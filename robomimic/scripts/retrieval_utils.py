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
from dtw_utils import DistanceResult, get_distance_matrix, compute_accumulated_cost_matrix_subsequence_dtw_21, compute_optimal_warping_path_subsequence_dtw_21, compare_distance_result
from functools import cmp_to_key


def retrieve_data(retrieval_task: str,
                  task_dataset_key: str,
                  retrieval_dataset_key: str,
                  retrieval_filter: str,
                  save_path: str,
                  seed: int,
                #   stack_length=5,
                  frame_stack=5,
                  seq_length=5,
                  n_demos = 3,
                  n_retrieve = 10,
                  demo_sub_traj_length = None,
                  demo_sub_traj_stride = None,
                  img_key="agentview_rgb",
                  embed_key = "model_class_facebook_dinov2-base_pooling_avg_model_DINOv2",
                  embed_img_keys = [
                        "eye_in_hand_rgb",
                        "agentview_rgb",
                    ],
                  debug=False,
                  chunks=None,
                  auto_slice=False
                  ):
    
    # seed the retrieval
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load task dataset
    task_dataset, task_dataset_paths, task_embed_paths = get_datasets(dataset_key=task_dataset_key, img_key=img_key, embed_key=embed_key, embed_img_keys=embed_img_keys)
    task_embeds = [np.mean(task_dataset[i]["embeds"], keepdims=True, axis=1) for i in range(len(task_dataset))]

    task_idcs = []
    for i in range(len(task_dataset)):
        task_idcs.append(retrieval_task in task_dataset[i]["meta"]["dataset_path"])
    task_idcs = np.where(task_idcs)[0]

    # load retrieval dataset
    retrieval_dataset, retrieval_dataset_paths, retrieval_embed_paths = get_datasets(dataset_key=retrieval_dataset_key, dataset_filter=retrieval_filter, img_key=img_key, embed_key=embed_key, embed_img_keys=embed_img_keys)
    retrieval_embeds = [np.mean(retrieval_dataset[i]["embeds"], keepdims=True, axis=1) for i in range(len(retrieval_dataset))]

    retrieval_idcs = []
    for i in range(len(retrieval_dataset)):
        retrieval_idcs.append(retrieval_task in retrieval_dataset[i]["meta"]["dataset_path"])
    retrieval_idcs = np.where(retrieval_idcs)[0]

    # get demo indicies
    demo_idcs = get_demo_indicies(task_idcs=task_idcs, dataset=task_dataset, n_demos=n_demos)
    
    # for i in demo_idcs:
    #     print(i)
    #     print(task_dataset[i]['meta'])
    

# # # DYNAMIC TIME WARPING

    # chunk demo embeddings
    if chunks is not None or auto_slice == True:
        demo_embeds, _ = slice_demos(task_dataset, demo_idcs, slices=chunks, auto=auto_slice)
    else:
        demo_embeds, _ = chunk_demos(task_dataset, demo_idcs, demo_sub_traj_length, demo_sub_traj_stride)
    
    # import IPython; IPython.embed()
    demo_embeds = [np.mean(demo_embed, keepdims=True, axis=1) for demo_embed in demo_embeds]

    # match demo to retrieval embeddings
    matches = []
    for demo_embed in demo_embeds:
        matches.append(get_matches_dtw(demo_embed, retrieval_embeds))
    
    # select top k matches (uniform across demos)
    top_k_matches = get_top_k_matches(matches, n_retrieve)
    matches_flat = flatten_2d_array(top_k_matches)


# # # BEHAVIOR/FLOW RETRIEVAL
    

    # # chunk demo embeddings to states

    # # concatenate + aggregate demo embeddings
    # demo_state = [np.mean(task_dataset[i]["embeds"], keepdims=True, axis=1) for i in demo_idcs]
    # demo_state_embeds = np.concatenate(demo_state, axis=0)

    
    # # concatenate + aggregate retrieval embeddings
    # state_embeds = []
    # state_meta = []
    # for i in range(len(retrieval_dataset)):
        
    #     # aggregate embeddings
    #     embeds_tmp = np.mean(retrieval_dataset[i]["embeds"], keepdims=True, axis=1)
    #     state_embeds.append(embeds_tmp)

    #     # extract meta info
    #     for l in range(len(embeds_tmp)):
    #         meta_tmp = retrieval_dataset[i]["meta"]
    #         # add index in dataset
    #         meta_tmp["demo_idx"] = i
    #         # add start and end idx
    #         meta_tmp["start_idx"] = l
    #         meta_tmp["end_idx"] = l+1
    #         state_meta.append(meta_tmp)

    # # concatenate embeddings
    # state_embeds = np.concatenate(state_embeds, axis=0)
    
    # # compute cosine sim
    # cosine_sim = get_matches_cosine(demo_state_embeds, state_embeds)

    # matches = []
    # from tqdm import trange
    # for i_demo in trange(cosine_sim.shape[0]):
    #     match = []
    #     # pre-filter top "n_retrieve" per sample to avoid creating 500k-dim array
    #     top_idcs = np.argpartition(-cosine_sim[i_demo], n_retrieve)[:n_retrieve]
        
    #     # create matches/results
    #     for top_idx in top_idcs:
    #         demo_idx = state_meta[top_idx]["demo_idx"] # index in dataset
    #         start_idx = state_meta[top_idx]["start_idx"] # start idx in traj
    #         end_idx = state_meta[top_idx]["end_idx"] # end idx in traj
    #         match.append(DistanceResult(demo_idx, start=start_idx, end=end_idx, cost=-cosine_sim[i_demo][top_idx], segment_matched_to=i_demo))

    #     matches.append(match)

    # # select top k matches (uniform across demos)
    # top_k_matches = get_top_k_matches(matches, n_retrieve)
    # matches_flat = flatten_2d_array(top_k_matches)

# # # 


    # generate demo results/"matches"
    demo_results = []
    for demo_idx in demo_idcs:
        demo_results.append(DistanceResult(demo_idx, start=0, end=len(task_dataset[demo_idx]["embeds"]), cost=0, segment_matched_to=-1))

    len_threshold = -np.inf # -np.inf -> no threshold

    dtw_dataset = LIBEROTrajDataset(None, img_key=img_key, img_size=(128,128),
                          embed_paths=[], embed_key=[], embed_img_keys=[],
                          n_samples=None, verbose=False)
    
    # set dataset_paths manually to avoid load()
    dtw_dataset.dataset_paths = retrieval_dataset_paths + task_dataset_paths

    def push_to_dataset(results, source_dataset, target_dataset):
            
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
                target_dataset._push(data_file, embed_file, k, j, start_idx, end_idx, load_imgs=False, load_embeds=False)

            # close files
            data_file.close()
            embed_file.close()
        
        return target_dataset

    # push demos
    push_to_dataset(demo_results, task_dataset, dtw_dataset)
    dtw_demo_idcs = np.arange(0, len(dtw_dataset), 1)

    # push sorted matches
    push_to_dataset(sorted(matches_flat, key=lambda x: x.cost), retrieval_dataset, dtw_dataset)
    dtw_idcs = np.arange(0, len(dtw_dataset), 1)

    dataset_path_out = os.path.join(DATASET_ROOT_DIR, save_path)
    os.makedirs(dataset_path_out, exist_ok=True)
    model_name = "dinov2"
    filename_out=f"{model_name}_{img_key}_demos_{n_demos}_dtw_{n_retrieve}.hdf5"
    
    # remove file if exists
    try:
        os.remove(os.path.join(dataset_path_out, filename_out))
        print(f"Removed existing file: {filename_out}")
    except:
        pass
    
    save_data(dtw_demo_idcs, dtw_dataset, dataset_path_out, filename_out=filename_out, frame_stack=frame_stack, seq_length=seq_length)
    
    return os.path.join(dataset_path_out, filename_out)


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


def get_matches_dtw(sub_trajectory, retrieval_dataset, demo_idcs=[]):
    
    results = []
    average_len = 0


    for i, traj in enumerate(tqdm(retrieval_dataset, desc="matching", disable=not TQDM)):

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
        results.append(DistanceResult(i, start=a_ast, end=b_ast + 1, cost=cost, segment_matched_to=i))

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
            demo_embeds.append(dataset[di]["embeds"][slice_idx[i]: slice_idx[i+1]])
            demo_i_to_segment.append(-1)
    
    return demo_embeds, demo_i_to_segment

def chunk_demos(dataset, demo_idcs, demo_sub_traj_length, demo_sub_traj_stride):
    demo_embeds = []
    demo_i_to_segment = []
    
    for di in demo_idcs:
        traj_len = len(dataset[di]["embeds"])
        length, stride = demo_sub_traj_length, demo_sub_traj_stride
        start_idx, end_idx = 0, length
        
        segment_id = 0
        
        if demo_sub_traj_length is None and demo_sub_traj_stride is None:
    #         demo_embeds.append(dataset_demo[di]["embeds"])
            demo_embeds.append(dataset[di]["embeds"])
            # demo_results.append(DistanceResult(index=di, start=0, end=len(dataset_demo[di]["embeds"]), cost=0))
            demo_i_to_segment.append(segment_id)
            segment_id += 1
            
            continue

        # make sure to include last segment even if len < length
        while end_idx < traj_len + stride:
            print(end_idx, traj_len, stride)
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
   
    
    

def get_datasets(dataset_key, img_key, embed_key, embed_img_keys, dataset_filter=None):
    
    dataset_registry_demo = {}
    for dr in [dataset_key]:
        dataset_registry_demo[dr] = DATASET_REGISTRY[dr]
    dataset_paths, embed_paths = get_dataset_paths(root_dir=DATASET_ROOT_DIR, dataset_registry=dataset_registry_demo, return_embed_paths=True, verbose=True)
    
    # remove converted datasets
    dataset_paths = np.array(dataset_paths)[["embeds" not in ds for ds in dataset_paths]].tolist()
    embed_paths = np.array(embed_paths)[["embeds_embeds" not in ds for ds in embed_paths]].tolist()

    # filter datasets
    if dataset_filter is not None:
        print(f"Filtering out {dataset_filter}")
        select = [dataset_filter not in ds for ds in dataset_paths]
        dataset_paths = np.array(dataset_paths)[select].tolist()
        embed_paths = np.array(embed_paths)[select].tolist()

    dataset = LIBEROTrajDataset(dataset_paths, img_key=None, img_size=(128,128), embed_paths=embed_paths, embed_key=embed_key, embed_img_keys=embed_img_keys, n_samples=None)
                            
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




def save_data(demo_idcs, dataset, dataset_path_out, filename_out, frame_stack, seq_length): 
    
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
                f_load.copy(f'data/{demo_key}', grp, name=new_demo_key)
                
               # crop traj to fit start_idx : end_idx
                start_idx = dataset[idx]["meta"]["start_idx"]
                end_idx = dataset[idx]["meta"]["end_idx"]


                extra_start = max(0, 0 - start_idx + frame_stack - 1) # we want to pad by 4 if the frame stack is 5
                extra_end = max(0, end_idx - len(grp[new_demo_key]["actions"]) + seq_length - 1)
                # deal with padding for the beginning
                start_idx = max(0, start_idx - frame_stack)
                end_idx = min(end_idx + seq_length, len(grp[new_demo_key]["actions"]))                

                
                try:
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
                except:
                    import IPython; IPython.embed()
                    
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
                grp[new_demo_key].attrs["ep_meta"] = json.dumps({"lang": dataset[idx]["lang"]})

        with h5py.File(dataset[demo_idcs[0]]["meta"]["dataset_path"], "r", swmr=True) as f_load:
            for attr_name, attr_value in f_load["data"].attrs.items():
                f_new["data"].attrs[attr_name] = attr_value
        
        demo_mask = np.array(demo_keys_mask, dtype="S")
        all_mask = np.array(list(f_new["data"].keys()), dtype="S")
        retrieval_mask = np.setdiff1d(all_mask, demo_mask)
        grp_mask.create_dataset("demos", data=demo_mask)
        grp_mask.create_dataset("retrieval", data=retrieval_mask)
        grp_mask.create_dataset("all", data=all_mask)


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
        sub_trajectories.append(states[start_idx:stop + 1])  # Add the segment
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
            merged_segments.append(current_segment)  # Save the segment if it's long enough
            current_segment = segments[i]  # Start a new segment
        
        prev_segment = current_segment
    
    # If the last segment is too short, merge it with the previous
    if len(current_segment) < min_length:
        merged_segments[-1] = np.vstack((merged_segments[-1], current_segment))
    else:
        merged_segments.append(current_segment)
    
    return merged_segments