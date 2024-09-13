import os
import numpy as np
from pathlib import Path

# DATASET REGISTRY
# add new datasets here
# recursively searches "dataset_path" for "file_name" while excluding paths including "exclude_path"
DATASET_REGISTRY = {
    "robocasa":{
        "dataset_path": "robocasa/v0.1/single_stage",
        "exclude_path": "_dist",
        "file_name": "demo_gentex_im128_randcams.hdf5",
    },
    "robocasa_no_pnp":{
        "dataset_path": "robocasa/v0.1/single_stage",
        "exclude_path": "PnP",
        "file_name": "demo_gentex_im128_randcams.hdf5",
    },
    "mimicgen":{
        "dataset_path": "mimicgen",
        "exclude_path": None,
        "file_name": "*.hdf5",
    },
    "mimicgen_converted":{
        "dataset_path": "mimicgen",
        "exclude_path": None,
        "file_name": "*_robocasa_demo.hdf5",
    },
    "robomimic":{
        "dataset_path": "robomimic",
        "exclude_path": None,
        "file_name": "image_v141.hdf5",
    },
    "robomimic_converted":{
        "dataset_path": "robomimic",
        "exclude_path": None,
        "file_name": "image_v141_robocasa_demo.hdf5",
    },
    "libero":{
        "dataset_path": "libero",
        "exclude_path": None,
        "file_name": "*.hdf5",
    },
    "libero_converted":{
        "dataset_path": "libero",
        "exclude_path": None,
        "file_name": "*_robocasa_demo.hdf5",
    },
    "libero_10_converted":{
        "dataset_path": "libero",
        "exclude_path": "libero_90",
        "file_name": "*_robocasa_demo.hdf5",
    },
    "libero_90_converted":{
        "dataset_path": "libero",
        "exclude_path": "libero_10",
        "file_name": "*_robocasa_demo.hdf5",
    },
    "viola":{
        "dataset_path": "viola",
        "exclude_path": None,
        "file_name": "robocasa_demo.hdf5"
    },
    "droid": {
        "dataset_path": "droid_100",
        "exclude_path": None,
        "file_name": "robocasa_demo.hdf5"
    },
    "libero_90": {
        "dataset_path": "libero_90",
        "exclude_path": "_embeds.hdf5",
        "file_name": "*demo.hdf5"
    },
    "libero_10": {
        "dataset_path": "libero_10",
        "exclude_path": "_embeds.hdf5",
        "file_name": "*demo.hdf5"
    },
        
}

# DATASET ROOT DIRECTORY
# change this to your dataset root directory
DATASET_ROOT_DIR = "/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/students/mem1pi/datasets/"

def get_dataset_paths(root_dir, dataset_registry, return_embed_paths=False, dataset_type=None, verbose=1):
    """
    Find all datasets in the root_dir based on a dataset_registry (dataset_paths, file_names).
    Args:
        root_dir (str): root directory to search for datasets
        dataset_registry (dict): dataset registry with dataset_path, exclude_path, file_name
        return_embed_paths (bool): also return embed_paths
        verbose (int): verbosity level
    """
    dataset_paths_all = []
    for name, dataset in dataset_registry.items():
        if dataset_type is not None and dataset_type != name:
            continue
        # find all datasets
        dataset_directory = Path(os.path.join(root_dir, dataset["dataset_path"]))
        dataset_paths = list(dataset_directory.rglob(dataset["file_name"]))

        # exclude certain datasets
        dataset_paths_tmp = []
        for pt in dataset_paths:
            if dataset["exclude_path"] is None or dataset["exclude_path"] not in str(pt):
                dataset_paths_tmp += [str(pt)]
        dataset_paths_all += dataset_paths_tmp
        if verbose:
            print(f"Found {len(dataset_paths_tmp)} datasets at {str(dataset_directory)} * {dataset['file_name']}")
    
    if return_embed_paths:
        embed_paths_all = [pt.replace(".hdf5", "_embeds.hdf5") for pt in dataset_paths_all]
        return dataset_paths_all, embed_paths_all
    else:
        return dataset_paths_all

def get_dataset_names(dataset_paths):
    dataset_names = []
    for dp in dataset_paths:
        if "libero" in dp:
            name = "_".join(dp.split("/")[-1].split(".")[0].split("SCENE")[1].split("_")[1:-3])
        elif "robomimic" in dp or "robocasa" in dp:
            name = dp.split("/")[-3]
        dataset_names.append(name)
    return np.array(dataset_names)