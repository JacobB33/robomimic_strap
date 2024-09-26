import os
import torch
import imageio
import numpy as np
from copy import copy
import h5py
import json
from tqdm import trange, tqdm

TQDM = False

# add explore_ret to path so it doesn't break:
import sys

sys.path.append("/home/mem1pi/projects/explore_ret")
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)


from hdf5_dataset import LIBEROTrajDataset
from data_utils import get_dataset_paths, DATASET_ROOT_DIR, DATASET_REGISTRY
from dtw_utils import (
    DistanceResult,
    get_distance_matrix,
    compute_accumulated_cost_matrix_subsequence_dtw_21,
    compute_optimal_warping_path_subsequence_dtw_21,
    compare_distance_result,
)
from functools import cmp_to_key
from mimic_retrieval_utils import (
    push_to_dataset,
    get_top_k_matches,
    flatten_2d_array,
    get_datasets,
    get_demo_indicies,
    save_data,
)


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

    # # DYNAMIC TIME WARPING

    # chunk demo embeddings
    if chunks is not None or auto_slice == True:
        demo_embeds, _ = slice_demos(
            task_dataset, demo_idcs, slices=chunks, auto=auto_slice
        )

    import IPython

    IPython.embed()
    demo_embeds = [
        np.mean(demo_embed, keepdims=True, axis=1) for demo_embed in demo_embeds
    ]

    # match demo to retrieval embeddings
    matches = []
    for demo_embed in demo_embeds:
        matches.append(get_matches_dtw(demo_embed, retrieval_embeds))

    # select top k matches (uniform across demos)
    top_k_matches = get_top_k_matches(matches, n_retrieve)
    matches_flat = flatten_2d_array(top_k_matches)

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


def get_matches_dtw(sub_trajectory, retrieval_dataset, demo_idcs=[]):

    results = []
    average_len = 0

    for i, traj in enumerate(
        tqdm(retrieval_dataset, desc="matching", disable=not TQDM)
    ):

        if i in demo_idcs:
            continue

        if len(traj) < len(sub_trajectory):
            # there can't be a valid match here so idk
            continue

        distance_matrix = get_distance_matrix(
            sub_trajectory[:, 0:1].squeeze(1), traj[:, 0:1].squeeze(1)
        )

        accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(
            distance_matrix
        )
        path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)

        a_ast = path[0, 1]
        if a_ast < 0:
            assert a_ast == -1
            a_ast = 0
        b_ast = path[-1, 1]
        cost = accumulated_cost_matrix[-1, b_ast]
        results.append(
            DistanceResult(
                i, start=a_ast, end=b_ast + 1, cost=cost, segment_matched_to=i
            )
        )

    return results


def slice_demos(dataset, demo_idcs, slices, auto=False):
    demo_embeds = []
    demo_i_to_segment = []

    # auto generate slices
    if auto:

        slices = {}
        for di in demo_idcs:

            # load ee_pos
            path = dataset[di]["meta"]["dataset_path"]
            dk = dataset[di]["meta"]["demo_key"]
            with h5py.File(path, "r", swmr=True) as file:
                states = file["data"][dk]["obs"]["ee_pos"][:]

            # segment using state derivative heuristic
            segments = segment_trajectory_by_derivative(states, threshold=5e-3)
            merged_segments = merge_short_segments(segments, min_length=20)

            # extract slice format
            seg_idcs = [0]
            for i, seg in enumerate(merged_segments):
                seg_idcs.append(seg_idcs[i] + len(seg))

            # remove 0
            slices[di] = seg_idcs[1:]

    print(slices)

    # slice and dice demos
    for di in demo_idcs:
        assert di in slices
        slice_idx = [0] + slices[di]
        for i in range(len(slice_idx) - 1):
            demo_embeds.append(dataset[di]["embeds"][slice_idx[i] : slice_idx[i + 1]])
            demo_i_to_segment.append(-1)

    return demo_embeds, demo_i_to_segment


def segment_trajectory_by_derivative(states, threshold=2.5e-3):

    # Calculate the absolute derivative of the first three states (X, Y, Z)
    diff = np.diff(states[:, :3], axis=0)
    abs_diff = np.sum(np.abs(diff), axis=1)

    # Find points where the derivative is below the threshold (indicating a stop)
    stops = np.where(abs_diff < threshold)[0]

    # Initialize the sub-trajectories list
    sub_trajectories = []
    start_idx = 0

    # Segment the trajectory at each stop point
    for stop in stops:
        sub_trajectories.append(states[start_idx : stop + 1])  # Add the segment
        start_idx = stop + 1  # Update start index

    # Append the last remaining segment
    if start_idx < len(states):
        sub_trajectories.append(states[start_idx:])

    return sub_trajectories


def merge_short_segments(segments, min_length=5):

    merged_segments = []
    current_segment = segments[0]

    for i in range(1, len(segments)):
        # If the current segment is too short, merge it with the next
        if len(current_segment) < min_length:
            current_segment = np.vstack((current_segment, segments[i]))
        else:
            merged_segments.append(
                current_segment
            )  # Save the segment if it's long enough
            current_segment = segments[i]  # Start a new segment

        prev_segment = current_segment

    # If the last segment is too short, merge it with the previous
    if len(current_segment) < min_length:
        merged_segments[-1] = np.vstack((merged_segments[-1], current_segment))
    else:
        merged_segments.append(current_segment)

    return merged_segments
