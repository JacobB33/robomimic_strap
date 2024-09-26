import os
import torch
import imageio
import numpy as np
from copy import copy
import h5py
import json
from tqdm import trange, tqdm

import sys

sys.path.append("/home/mem1pi/projects/explore_ret")
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)

from hdf5_dataset import LIBEROTrajDataset
from dtw_utils import DistanceResult
from functools import cmp_to_key
from data_utils import get_dataset_paths, DATASET_ROOT_DIR, DATASET_REGISTRY

TQDM = True


def push_to_dataset(results, source_dataset, target_dataset, len_threshold=-np.inf):

    for result in tqdm(results):

        # open files
        dataset_path = source_dataset[result.index]["meta"]["dataset_path"]
        data_file = h5py.File(dataset_path, "r")
        embed_file = h5py.File(source_dataset[result.index]["meta"]["embed_path"], "r")

        # get keys, label, start and end
        k = source_dataset[result.index]["meta"]["demo_key"]
        j = target_dataset.dataset_paths.index(dataset_path)
        start_idx = result.start
        end_idx = result.end

        # save if larger than thresh
        if end_idx is not None and end_idx - start_idx >= len_threshold:
            # don't load images and embeddings
            target_dataset._push(
                data_file,
                embed_file,
                k,
                j,
                start_idx,
                end_idx,
                load_imgs=False,
                load_embeds=False,
            )

        # close files
        data_file.close()
        embed_file.close()

    return target_dataset


def get_top_k_matches(matches, n_retrieve):
    k = int(n_retrieve / len(matches))
    top_k_matches = []

    for match in matches:
        match = sorted(match)
        top_k_matches.append(match[:k])

    return top_k_matches


def flatten_2d_array(arr_2d):
    flattened = []
    for sublist in arr_2d:
        flattened.extend(sublist)
    return flattened


def get_datasets(dataset_key, img_key, embed_key, embed_img_keys, dataset_filter=None):

    dataset_registry_demo = {}
    for dr in [dataset_key]:
        dataset_registry_demo[dr] = DATASET_REGISTRY[dr]
    dataset_paths, embed_paths = get_dataset_paths(
        root_dir=DATASET_ROOT_DIR,
        dataset_registry=dataset_registry_demo,
        return_embed_paths=True,
        verbose=True,
    )

    # remove converted datasets
    dataset_paths = np.array(dataset_paths)[
        ["embeds" not in ds for ds in dataset_paths]
    ].tolist()
    embed_paths = np.array(embed_paths)[
        ["embeds_embeds" not in ds for ds in embed_paths]
    ].tolist()

    # filter datasets
    if dataset_filter is not None:
        print(f"Filtering out {dataset_filter}")
        select = [dataset_filter not in ds for ds in dataset_paths]
        dataset_paths = np.array(dataset_paths)[select].tolist()
        embed_paths = np.array(embed_paths)[select].tolist()

    dataset = LIBEROTrajDataset(
        dataset_paths,
        img_key=None,
        img_size=(128, 128),
        embed_paths=embed_paths,
        embed_key=embed_key,
        embed_img_keys=embed_img_keys,
        n_samples=None,
    )

    return dataset, dataset_paths, embed_paths


def get_demo_indicies(task_idcs, dataset, n_demos):
    # search dataset for task samples
    # in case of sub-trexplore_ret

    demo_start_idcs = {}
    # TODO: this is very redundant, why is it this bad
    for task_idx in task_idcs:
        demo_key = dataset[task_idx]["meta"]["demo_key"]
        demo_start_idcs[demo_key] = {"start": task_idx, "end": task_idx + 1}

    demo_keys = np.random.choice(
        list(demo_start_idcs.keys()), size=n_demos, replace=False
    )

    demo_idcs = []
    for demo_key in demo_keys:
        demo_idcs.append(
            np.arange(
                demo_start_idcs[demo_key]["start"], demo_start_idcs[demo_key]["end"], 1
            )
        )
    demo_idcs = np.concatenate(demo_idcs)
    return demo_idcs


def save_data(
    demo_idcs, dataset, dataset_path_out, filename_out, frame_stack, seq_length
):

    # create new file
    with h5py.File(os.path.join(dataset_path_out, filename_out), "w") as f_new:

        # CREATE
        grp = f_new.create_group("data")
        demo_keys_mask = []
        grp_mask = f_new.create_group("mask")

        # DEMOS
        for idx in trange(len(dataset), desc="saving", disable=not TQDM):

            # load old file
            dataset_path = dataset[idx]["meta"]["dataset_path"]
            with h5py.File(dataset_path, "r", swmr=True) as f_load:
                # get demo key
                demo_key = dataset[idx]["meta"]["demo_key"]
                if type(demo_key) != str:
                    demo_key = demo_key.decode("utf-8")

                # re-map demo key
                new_demo_key = f"demo_{idx}"

                if idx in demo_idcs:
                    demo_keys_mask.append(new_demo_key)
                # copy demo from old file to new file
                f_load.copy(f"data/{demo_key}", grp, name=new_demo_key)

                # crop traj to fit start_idx : end_idx
                start_idx = dataset[idx]["meta"]["start_idx"]
                end_idx = dataset[idx]["meta"]["end_idx"]

                extra_start = max(
                    0, 0 - start_idx + frame_stack - 1
                )  # we want to pad by 4 if the frame stack is 5
                extra_end = max(
                    0, end_idx - len(grp[new_demo_key]["actions"]) + seq_length - 1
                )
                # deal with padding for the beginning
                start_idx = max(0, start_idx - frame_stack)
                end_idx = min(end_idx + seq_length, len(grp[new_demo_key]["actions"]))

                try:
                    for lk in ["actions", "states"]:
                        tmp_copy = np.array(
                            grp[new_demo_key][lk][start_idx:end_idx]
                        ).copy()
                        # pad the start if needed
                        if extra_start:
                            tmp_copy = np.concatenate(
                                [
                                    np.stack(
                                        [tmp_copy[0] for i in range(extra_start)],
                                        axis=0,
                                    ),
                                    tmp_copy,
                                ],
                                axis=0,
                            )
                        # pad the end if needed
                        if extra_end:
                            tmp_copy = np.concatenate(
                                [
                                    tmp_copy,
                                    np.stack(
                                        [tmp_copy[-1] for i in range(extra_end)], axis=0
                                    ),
                                ],
                                axis=0,
                            )
                        del grp[new_demo_key][lk]
                        grp[new_demo_key][lk] = tmp_copy
                except:
                    import IPython

                    IPython.embed()

                for lk in grp[new_demo_key]["obs"].keys():
                    tmp_copy = np.array(
                        grp[new_demo_key]["obs"][lk][start_idx:end_idx]
                    ).copy()
                    if extra_start:
                        tmp_copy = np.concatenate(
                            [
                                np.stack(
                                    [tmp_copy[0] for i in range(extra_start)], axis=0
                                ),
                                tmp_copy,
                            ],
                            axis=0,
                        )
                    # pad the end if needed
                    if extra_end:
                        tmp_copy = np.concatenate(
                            [
                                tmp_copy,
                                np.stack(
                                    [tmp_copy[-1] for i in range(extra_end)], axis=0
                                ),
                            ],
                            axis=0,
                        )
                    del grp[new_demo_key]["obs"][lk]
                    grp[new_demo_key]["obs"][lk] = tmp_copy

                for attr_name, attr_value in f_load[f"data/{demo_key}"].attrs.items():
                    grp[new_demo_key].attrs[attr_name] = attr_value

                grp[new_demo_key].attrs["num_samples"] = len(
                    grp[new_demo_key]["actions"]
                )
                grp[new_demo_key].attrs["ep_meta"] = json.dumps(
                    {"lang": dataset[idx]["lang"]}
                )

        with h5py.File(
            dataset[demo_idcs[0]]["meta"]["dataset_path"], "r", swmr=True
        ) as f_load:
            for attr_name, attr_value in f_load["data"].attrs.items():
                f_new["data"].attrs[attr_name] = attr_value

        demo_mask = np.array(demo_keys_mask, dtype="S")
        all_mask = np.array(list(f_new["data"].keys()), dtype="S")
        retrieval_mask = np.setdiff1d(all_mask, demo_mask)
        grp_mask.create_dataset("demos", data=demo_mask)
        grp_mask.create_dataset("retrieval", data=retrieval_mask)
        grp_mask.create_dataset("all", data=all_mask)
