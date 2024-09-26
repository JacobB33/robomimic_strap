import os
import torch
import imageio
import numpy as np
from copy import copy
import h5py
import json
from tqdm import trange, tqdm

TQDM = True

# add explore_ret to path so it doesn't break:
import sys

sys.path.append("/home/mem1pi/projects/explore_ret")
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)

from hdf5_dataset import LIBEROTrajDataset
from dtw_utils import DistanceResult

from mimic_retrieval_utils import (
    push_to_dataset,
    get_top_k_matches,
    flatten_2d_array,
    get_datasets,
    get_demo_indicies,
    save_data,
)
from data_utils import get_dataset_paths, DATASET_ROOT_DIR, DATASET_REGISTRY


def retrieve_data(
    retrieval_task: str,
    task_dataset_key: str,
    retrieval_dataset_key: str,
    retrieval_filter: str,
    save_path: str,
    seed: int,
    frame_stack=5,
    seq_length=5,
    n_demos=3,
    n_retrieve=10,
    demo_sub_traj_length=None,
    demo_sub_traj_stride=None,
    img_key="agentview_rgb",
    embed_key="model_class_facebook_dinov2-base_pooling_avg_model_DINOv2",
    embed_img_keys=[
        "eye_in_hand_rgb",
        "agentview_rgb",
    ],
    debug=False,
    chunks=None,
    auto_slice=False,
):

    # seed the retrieval
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load task dataset
    task_dataset, task_dataset_paths, task_embed_paths = get_datasets(
        dataset_key=task_dataset_key,
        img_key=img_key,
        embed_key=embed_key,
        embed_img_keys=embed_img_keys,
    )
    task_embeds = [
        np.mean(task_dataset[i]["embeds"], keepdims=True, axis=1)
        for i in range(len(task_dataset))
    ]

    task_idcs = []
    for i in range(len(task_dataset)):
        task_idcs.append(retrieval_task in task_dataset[i]["meta"]["dataset_path"])
    task_idcs = np.where(task_idcs)[0]

    # load retrieval dataset
    retrieval_dataset, retrieval_dataset_paths, retrieval_embed_paths = get_datasets(
        dataset_key=retrieval_dataset_key,
        dataset_filter=retrieval_filter,
        img_key=img_key,
        embed_key=embed_key,
        embed_img_keys=embed_img_keys,
    )
    retrieval_embeds = [
        np.mean(retrieval_dataset[i]["embeds"], keepdims=True, axis=1)
        for i in range(len(retrieval_dataset))
    ]

    retrieval_idcs = []
    for i in range(len(retrieval_dataset)):
        retrieval_idcs.append(
            retrieval_task in retrieval_dataset[i]["meta"]["dataset_path"]
        )
    retrieval_idcs = np.where(retrieval_idcs)[0]

    # get demo indicies
    demo_idcs = get_demo_indicies(
        task_idcs=task_idcs, dataset=task_dataset, n_demos=n_demos
    )

    # for i in demo_idcs:
    #     print(i)
    #     print(task_dataset[i]['meta'])

    # # BEHAVIOR/FLOW RETRIEVAL

    # chunk demo embeddings to states

    # concatenate + aggregate demo embeddings
    demo_state = [
        np.mean(task_dataset[i]["embeds"], keepdims=True, axis=1) for i in demo_idcs
    ]
    demo_state_embeds = np.concatenate(demo_state, axis=0)

    # concatenate + aggregate retrieval embeddings
    state_embeds = []
    state_meta = []
    for i in range(len(retrieval_dataset)):

        # aggregate embeddings
        embeds_tmp = np.mean(retrieval_dataset[i]["embeds"], keepdims=True, axis=1)
        state_embeds.append(embeds_tmp)

        # extract meta info
        for l in range(len(embeds_tmp)):
            meta_tmp = retrieval_dataset[i]["meta"]
            # add index in dataset
            meta_tmp["demo_idx"] = i
            # add start and end idx
            meta_tmp["start_idx"] = l
            meta_tmp["end_idx"] = l + 1
            state_meta.append(meta_tmp)

    # concatenate embeddings
    state_embeds = np.concatenate(state_embeds, axis=0)

    # compute cosine sim
    cosine_sim = get_matches_cosine(demo_state_embeds, state_embeds)

    matches = []
    from tqdm import trange

    for i_demo in trange(cosine_sim.shape[0]):
        match = []
        # pre-filter top "n_retrieve" per sample to avoid creating 500k-dim array
        top_idcs = np.argpartition(-cosine_sim[i_demo], n_retrieve)[:n_retrieve]

        # create matches/results
        for top_idx in top_idcs:
            demo_idx = state_meta[top_idx]["demo_idx"]  # index in dataset
            start_idx = state_meta[top_idx]["start_idx"]  # start idx in traj
            end_idx = state_meta[top_idx]["end_idx"]  # end idx in traj
            match.append(
                DistanceResult(
                    demo_idx,
                    start=start_idx,
                    end=end_idx,
                    cost=-cosine_sim[i_demo][top_idx],
                    segment_matched_to=i_demo,
                )
            )

        matches.append(match)

    # import IPython; IPython.embed()
    # select top k matches (uniform across demos)
    top_k_matches = get_top_k_matches(matches, n_retrieve)
    matches_flat = flatten_2d_array(top_k_matches)
    print(f"Number of states retrieved: {len(matches_flat)}")
    assert len(matches_flat) > 0

    # # #

    # generate demo results/"matches"
    demo_results = []
    for demo_idx in demo_idcs:
        demo_results.append(
            DistanceResult(
                demo_idx,
                start=0,
                end=len(task_dataset[demo_idx]["embeds"]),
                cost=0,
                segment_matched_to=-1,
            )
        )

    len_threshold = -np.inf  # -np.inf -> no threshold

    dtw_dataset = LIBEROTrajDataset(
        None,
        img_key=img_key,
        img_size=(128, 128),
        embed_paths=[],
        embed_key=[],
        embed_img_keys=[],
        n_samples=None,
        verbose=False,
    )

    # set dataset_paths manually to avoid load()
    dtw_dataset.dataset_paths = retrieval_dataset_paths + task_dataset_paths

    # push demos
    push_to_dataset(demo_results, task_dataset, dtw_dataset)
    dtw_demo_idcs = np.arange(0, len(dtw_dataset), 1)

    # push sorted matches
    push_to_dataset(
        sorted(matches_flat, key=lambda x: x.cost), retrieval_dataset, dtw_dataset
    )
    dtw_idcs = np.arange(0, len(dtw_dataset), 1)

    dataset_path_out = os.path.join(DATASET_ROOT_DIR, save_path)
    os.makedirs(dataset_path_out, exist_ok=True)
    model_name = embed_key.split("_")[-1]
    filename_out = f"{model_name}_{img_key}_demos_{n_demos}_dtw_{n_retrieve}.hdf5"

    # remove file if exists
    try:
        os.remove(os.path.join(dataset_path_out, filename_out))
        print(f"Removed existing file: {filename_out}")
    except:
        pass

    save_data(
        dtw_demo_idcs,
        dtw_dataset,
        dataset_path_out,
        filename_out=filename_out,
        frame_stack=frame_stack,
        seq_length=seq_length,
    )

    return os.path.join(dataset_path_out, filename_out)


def get_matches_cosine(reference, features):

    # reference Nx1xD -> N # demo states
    # features Mx1xD -> M # retrieval states

    # check if reference has leading batch dim
    if reference.ndim != features.ndim:
        reference = np.expand_dims(reference, axis=0)

    # normalize the reference and features
    ref_norm = np.linalg.norm(reference, axis=2, keepdims=True)
    feat_norm = np.linalg.norm(features, axis=2, keepdims=True)

    ref_embed = reference / ref_norm
    feat_embed = features / feat_norm

    # compute cosine similarity
    cosine_sim = np.matmul(ref_embed.squeeze(), feat_embed.squeeze().T)

    # returns NxM
    return cosine_sim
