import os
import torch
import imageio
import numpy as np
from copy import copy
import h5py

# add explore_ret to path so it doesn't break:
import sys
sys.path.append("/gscratch/weirdlab/jacob33/retrieval/explore_ret")
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)


from hdf5_dataset import HDF5TrajDataset, LIBEROTrajDataset
from data_utils import get_dataset_paths, DATASET_ROOT_DIR, DATASET_REGISTRY
from dtw_utils import DistanceResult, get_distance_matrix, compute_accumulated_cost_matrix_subsequence_dtw_21, compute_optimal_warping_path_subsequence_dtw_21, compare_distance_result
from functools import cmp_to_key


def retrieve_data(retrieval_task: str,
                  save_path: str,
                  seed: int,
                  stack_length=5,
                  n_demos = 3,
                  demo_sub_traj_length = 25,
                  demo_sub_traj_stride = 25,
                  img_key="agentview_rgb",
                  embed_key = "model_class_facebook_dinov2-base_pooling_avg_model_DINOv2",
                  embed_img_keys = [
                        "eye_in_hand_rgb",
                        "agentview_rgb",
                    ],
                  k=1000):
    # seed the retrieval
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load the data
    dataset, dataset_paths, embed_paths = get_datasets(img_key=img_key, embed_key=embed_key, embed_img_keys=embed_img_keys)
    images = dataset[:]["imgs"]
    embeds = [np.mean(dataset[i]["embeds"], keepdims=True, axis=1) for i in range(len(dataset))]

    # get the expert indexes
    libero_10_idcs = np.where(["libero_10" in dataset[i]["meta"]["dataset_path"] for i in range(len(dataset))])[0]
    task_idcs = np.where([retrieval_task in dataset[i]["meta"]["dataset_path"] for i in range(len(dataset))])[0]
    demo_idcs = get_demo_indicies(task_idcs=task_idcs, dataset=dataset, n_demos=n_demos)
    
    # get the demo_embeds
    demo_embeds, demo_i_to_segment = chunk_demos(dataset, demo_idcs, demo_sub_traj_length, demo_sub_traj_stride)
    matches = []
    for id, demo_embed in zip(demo_i_to_segment, demo_embeds):
        matches.extend(get_matches(demo_embed, embeds, libero_10_idcs, id=id))
    
    top_k_matches = [[] for _ in range(max(demo_i_to_segment) + 1)]
    for match in matches:
        top_k_matches[match.segment_matched_to].append(copy(match))
    for i in range(len(top_k_matches)):
        top_k_matches[i] = sorted(top_k_matches[i])

    # merge each segment but not cross segment 
    for segment_part in top_k_matches:
        # check if any matches have overlapping indices
        segment_part.sort(key=cmp_to_key(compare_distance_result))
        i = 0
        while i < len(segment_part) -1:
            if segment_part[i].index != segment_part[i+1].index:
                i += 1
            elif segment_part[i].end >= segment_part[i+1].start:
                old_traj = segment_part.pop(i+1)
                segment_part[i].end = max(segment_part[i].end, old_traj.end)
                segment_part[i].cost = min(segment_part[i].cost, old_traj.cost) # Since this cost is now broken
            else:
                i += 1
        segment_part.sort(key= lambda x: x.cost)
        
    # Get the top k for each sgement 
    for i, match in enumerate(top_k_matches):
        index, frames = 0, 0
        while frames < k and i < len(match):
            frames += match[i].end - match[i].start
            index += 1
        top_k_matches[i] = match[:index]

    merged_matches = flatten_2d_array(top_k_matches)

    
    # This is code to add in the expert matches and then set their indicies:
    match_len = len(merged_matches)
    for demo_idx in demo_idcs:
        merged_matches.append(DistanceResult(demo_idx, start=0, end=len(dataset[demo_idx]["imgs"]), cost=0, segment_matched_to=-1))
    dtw_demo_idcs = [i for i in range(match_len, match_len + len(demo_idcs))]

    dtw_dataset = LIBEROTrajDataset(None, img_key=img_key, img_size=(128,128),
                          embed_paths=embed_paths, embed_key=embed_key, embed_img_keys=embed_img_keys,
                          n_samples=None)
    
    # Set here to avoid the load
    dtw_dataset.dataset_paths = dataset_paths

    # create dataset to push from
    for m in merged_matches:
        dataset_path = dataset[m.index]["meta"]["dataset_path"]
        data_file = h5py.File(dataset_path, "r")
        embed_file = h5py.File(dataset[m.index]["meta"]["embed_path"], "r")
        
        key = dataset[m.index]["meta"]["demo_key"]
        j = dataset_paths.index(dataset_path)

        dtw_dataset._push(data_file, embed_file, key, j, m.start, m.end, load=True)
        
        data_file.close()
        embed_file.close()
    
    dataset_path_out = os.path.join(DATASET_ROOT_DIR, save_path)
    os.makedirs(dataset_path_out, exist_ok=True)
    model_name = "dinov2"
    filename_out=f"{model_name}_{img_key}_{k}_dtw.hdf5"
    save_data(dtw_demo_idcs, dtw_dataset, dataset_path_out, filename_out=filename_out, stack_length=stack_length)
    return os.path.join(dataset_path_out, filename_out)

def flatten_2d_array(arr_2d):
    flattened = []
    for sublist in arr_2d:
        flattened.extend(sublist)
    return flattened
   
def get_matches(sub_trajectory, retrieval_dataset, demo_idcs, id):
    
    results = []
    average_len = 0


    for i, traj in enumerate(retrieval_dataset):

        if i in demo_idcs:
            continue
        
        if len(traj) < len(sub_trajectory):
            # there can't be a valid match here so idk
            continue
            
        distance_matrix = get_distance_matrix(
            sub_trajectory[:, 0:1].squeeze(1),
            traj[:, 0:1].squeeze(1)
        )
        accumulated_cost_matrix = compute_accumulated_cost_matrix_subsequence_dtw_21(distance_matrix)
        path = compute_optimal_warping_path_subsequence_dtw_21(accumulated_cost_matrix)

        a_ast = path[0, 1]
        if a_ast < 0:
            assert a_ast == -1
            a_ast = 0
        b_ast = path[-1, 1]
        cost = accumulated_cost_matrix[-1, b_ast]
        results.append(DistanceResult(i, start=a_ast, end=b_ast + 1, cost=cost, segment_matched_to=id))
    return results
  
    
def chunk_demos(dataset, demo_idcs, demo_sub_traj_length, demo_sub_traj_stride):
    demo_embeds = []
    demo_i_to_segment = []
    for di in demo_idcs:
        traj_len = len(dataset[di]["imgs"])
        length, stride = demo_sub_traj_length, demo_sub_traj_stride
        start_idx, end_idx = 0, length
        segment_id = 0
        # make sure to include last segment even if len < length
        while end_idx < traj_len + stride:
            if end_idx > traj_len:
                end_idx = traj_len
                
            # ensure minimum segment length
            if end_idx - start_idx >= 2:
                demo_embeds.append(dataset[di]["embeds"][start_idx:end_idx])
                demo_i_to_segment.append(segment_id)
            
            segment_id += 1
            start_idx += stride
            end_idx += stride
    return demo_embeds, demo_i_to_segment
   
    
    

def get_datasets(img_key, embed_key, embed_img_keys):
    
    dataset_paths, embed_paths = get_dataset_paths(root_dir=DATASET_ROOT_DIR, dataset_registry=DATASET_REGISTRY, return_embed_paths=True)
    dataset = LIBEROTrajDataset(dataset_paths, img_key=img_key, img_size=(128,128),
                            embed_paths=embed_paths, embed_key=embed_key, embed_img_keys=embed_img_keys,
                            n_samples=None)
    return dataset, dataset_paths, embed_paths


def get_demo_indicies(task_idcs, dataset, n_demos):
    # search dataset for task samples
    # in case of sub-trexplore_ret
    demo_start_idcs = {}
    # TODO: this is very redundant, why is it this bad
    for task_idx in task_idcs:
        demo_key = dataset[task_idx]["meta"]["demo_key"]
        demo_start_idcs[demo_key] = {"start": task_idx, "end": task_idx + 1}
        
    demo_keys = np.random.choice(list(demo_start_idcs.keys()), size=n_demos, replace=False)

    demo_idcs = []
    for demo_key in demo_keys:
        demo_idcs.append(np.arange(demo_start_idcs[demo_key]["start"], demo_start_idcs[demo_key]["end"], 1))
    demo_idcs = np.concatenate(demo_idcs)
    return demo_idcs




def save_data(demo_idcs, dataset, dataset_path_out, filename_out, stack_length): 
    
    # create new file
    with h5py.File(os.path.join(dataset_path_out, filename_out), "w") as f_new:

        # CREATE
        grp = f_new.create_group("data")

        # DEMOS
        for idx in range(len(dataset)):
        
            # load old file
            dataset_path = dataset[idx]["meta"]["dataset_path"]
            with h5py.File(dataset_path, "r") as f_load:
                # get demo key
                demo_key = dataset[idx]["meta"]["demo_key"]
                if type(demo_key) != str: 
                    demo_key = demo_key.decode("utf-8")
                    
                # re-map demo key
                new_demo_key = f"demo_{idx}"
                
                # copy demo from old file to new file
                f_load.copy(f'data/{demo_key}', grp, name=new_demo_key)
                
                
                # crop traj to fit start_idx : end_idx
                start_idx = dataset[idx]["meta"]["start_idx"]
                end_idx = dataset[idx]["meta"]["end_idx"]
                
                extra_start = max(0, 0 - start_idx + stack_length - 1) # we want to pad by 4 if the frame stack is 5
                extra_end = max(0, end_idx - len(grp[new_demo_key]["actions"]) + stack_length - 1)
                # deal with padding for the beginning
                start_idx = max(0, start_idx - stack_length)
                end_idx = min(end_idx + stack_length, len(grp[new_demo_key]["actions"]))
                
               
                
                
                for lk in ["actions", "states"]:
                    tmp_copy = np.array(grp[new_demo_key][lk][start_idx:end_idx]).copy()
                    # pad the start if needed
                    if extra_start:
                        tmp_copy = np.concatenate([np.stack([tmp_copy[0] for i in range(extra_start)], axis=0), tmp_copy], axis=0)
                    # pad the end if needed
                    if extra_end:
                        tmp_copy = np.concatenate([tmp_copy, np.stack([tmp_copy[-1] for i in range(extra_end)], axis=0)], axis=0)
                    del grp[new_demo_key][lk]
                    grp[new_demo_key][lk] = tmp_copy

                for lk in grp[new_demo_key]["obs"].keys():
                    tmp_copy = np.array(grp[new_demo_key]["obs"][lk][start_idx:end_idx]).copy()
                    if extra_start:
                        tmp_copy = np.concatenate([np.stack([tmp_copy[0] for i in range(extra_start)], axis=0), tmp_copy], axis=0)
                    # pad the end if needed
                    if extra_end:
                        tmp_copy = np.concatenate([tmp_copy, np.stack([tmp_copy[-1] for i in range(extra_end)], axis=0)], axis=0)
                    del grp[new_demo_key]["obs"][lk]
                    grp[new_demo_key]["obs"][lk] = tmp_copy
                
               
                for attr_name, attr_value in f_load[f'data/{demo_key}'].attrs.items():
                    grp[new_demo_key].attrs[attr_name] = attr_value
                
                grp[new_demo_key].attrs["num_samples"] = len(grp[new_demo_key]["actions"])

        with h5py.File(dataset[demo_idcs[0]]["meta"]["dataset_path"], "r") as f_load:
            for attr_name, attr_value in f_load["data"].attrs.items():
                f_new["data"].attrs[attr_name] = attr_value
   