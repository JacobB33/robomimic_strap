import time
import submitit_configs as sc
import submitit_tools as st
import submitit
from dataclasses import dataclass, field
import subprocess
import os
from argparse import Namespace
from tqdm import tqdm
from robomimic.macros import PERSON
if PERSON == "marius":
    DATASET_ROOT_DIR = "/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/students/mem1pi/retrieval_logs"
elif PERSON == "jacob":
    DATASET_ROOT_DIR = "/gscratch/weirdlab/jacob33/retrieval/robocasa/datasets"
else:
    assert False
DEBUG = False
SLURM = False

TASK_DATASET_KEY = "libero_10"
RETRIEVAL_DATASET_KEY = "libero_90"
WANDB_PROJECT_NAME = "multitask"

names = [
    # "mugs_state",
    # "book_state"
    # "mugs",
    # "book"
    "bowl_drawer",
    "mug_microwave",
    "moka_pots",
    "soup_sauce"
]

task_names = [
    # "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    # "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
    
    "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "put_both_moka_pots_on_the_stove",
    "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
]
task_filters = [
    None,
    None,
    None,
    None
    # "LIVING_ROOM_SCENE1",
    # "KITCHEN_SCENE8",
    # "STUDY_SCENE4"
]

retrieval_seeds = [
    0
]

train_seeds = [
    1234
]

policy = "transformer" # "transformer", "diffusion"
masks = ["demos", "all"] # "demos", "co_train"

n_demos = [5]
n_retrieve = [10, 100] # [10 * 50, 100 * 50] # 

auto_slice=True
chunks = None

@dataclass
class RobomimicTrainConfig(sc.BaseJobConfig):
    name: str = "dummy"
    
    policy: str = "dummy"
    filter_key: str = "dummy"

    n_demos: int = 5
    n_retrieve: int = 10

    retrieval_task: str  = "dummy"
    retrieval_filter: str = "dummy"
    
    dataset_save_path: str = "dummy"
    
    retrieval_seed: int = -1
    train_seed: int = -1

job_configs = []

for name, task_name, task_filter in zip(names, task_names, task_filters):

    for ret_seed in retrieval_seeds:
        for tr_seed in train_seeds:
            
            for nd in n_demos:
                for nr in n_retrieve:

                    specific_name = f"{name}_nd_{nd}_nr_{nr}_s_{ret_seed}_{tr_seed}"
                    job_configs.append(
                        RobomimicTrainConfig(
                            name=specific_name,

                            policy=policy,

                            n_demos=nd,
                            n_retrieve=nr,

                            retrieval_task=task_name, 
                            retrieval_filter=task_filter,

                            dataset_save_path=os.path.join(DATASET_ROOT_DIR, "retrieval", task_name, specific_name),
                            checkpoint_path=os.path.join(DATASET_ROOT_DIR, "retrieval", task_name, specific_name),
                            
                            retrieval_seed=ret_seed,
                            train_seed=tr_seed)
                        )
    
    
config = sc.SubmititExecutorConfig(root_folder="/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/students/mem1pi/datasets/retrieval/dtw",
                                slurm_name="retrieval",
                                slurm_partition="gpu-a40",
                                timeout_min=60 * 8,
                                cpus_per_task=6,
                                mem_gb=100,
                                slurm_gpus_per_node="a40:1",
                                )



class TrainingJob(st.BaseJob):
    def __init__(self, job_config: RobomimicTrainConfig, wandb_config):
        super().__init__(job_config, wandb_config)
        self.job_config: RobomimicTrainConfig = job_config

    def _job_call(self):
        import robomimic
        # import training script/function
        from robomimic.scripts.train import main
        from robomimic.scripts.retrieval_utils import retrieve_data


        if self.job_config.policy == "transformer":
            frame_stack = 5
            seq_length = 5
        elif self.job_config.policy == "diffusion":
            frame_stack = 2
            seq_length = 15
        
        # do retrieval!
        retrieved_data_save_path = retrieve_data(
            
            retrieval_task=self.job_config.retrieval_task,
            task_dataset_key=TASK_DATASET_KEY,
            retrieval_dataset_key=RETRIEVAL_DATASET_KEY,
            retrieval_filter=self.job_config.retrieval_filter,
            
            save_path=self.job_config.dataset_save_path,
            seed=self.job_config.retrieval_seed,

            frame_stack=frame_stack,
            seq_length=seq_length,
            
            n_demos = self.job_config.n_demos,
            n_retrieve = self.job_config.n_retrieve,
            
            demo_sub_traj_length = None,
            demo_sub_traj_stride = None,
            
            img_key="agentview_rgb",
            embed_key = "model_class_facebook_dinov2-base_pooling_avg_model_DINOv2",
            embed_img_keys = [
                    "eye_in_hand_rgb",
                    "agentview_rgb",
                ],
            
            debug=DEBUG,
            chunks=chunks,
            auto_slice=auto_slice
        )
        
        config_paths = []
        for mask in masks:
            
            if mask == "co_train":
                assert self.job_config.policy == "diffusion"

            if self.job_config.policy == "transformer":
                config_gen = "bc_xfmr_gen.py"
            elif self.job_config.policy == "diffusion":
                config_gen = "bc_diffusion_gen.py"

            if mask == "demos":
                co_training_tmp = False
            elif mask == "all":
                co_training_tmp = False
            elif mask == "co_train":
                co_training_tmp = True
                mask = None
            else:
                raise ValueError(f"Invalid mask: {mask} or co_training: {co_training_tmp}")

            # We don't want to pad the demos since they are already padded in our dataset
            cmd = f"python {os.path.dirname(robomimic.__file__)}/scripts/config_gen/{config_gen} --name {self.job_config.name}_{mask} --train_task {retrieved_data_save_path} --env libero --no_pad --file {f'--filter_key {mask}' if mask is not None else ''} {'--co_train' if co_training_tmp else ''} {'--debug' if DEBUG else ''}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)    
            # Parse the output to extract the config file path
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.startswith("python") and "--config" in line:
                    config_path = line.split("--config")[-1].strip().split()[0]
            
            args = Namespace(
                    config=config_path,
                    algo=None, 
                    name=None,
                    dataset=None,
                    co_dataset=None,
                    filter_key=None,
                    dataset_json=None,
                    ckpt_path=None,
                    output_dir=None,
                    seed=self.job_config.train_seed,
                    debug=False,
                    wandb_name=WANDB_PROJECT_NAME,
                )
            print(config_path)
            config_paths.append(config_path)
            if SLURM:
                main(args=args)
        return "done" if SLURM else config_paths
    
    def _initialize(self):
        pass

if not SLURM:
    configs = []
    for job_config in tqdm(job_configs, desc="gen configs + datasets"):
        job = TrainingJob(job_config, None)
        configs.extend(job._job_call())

    from robomimic.scripts.lfs_gen import generate_lfs_bsub_script
    generate_lfs_bsub_script(
        configs,
        time="16:00",
        cpus=8,
        mem=8000,
        gpus=1,
        queue="batch_v100",
        conda_env="libero_plain"
    )

else:
    state = st.SubmititState(TrainingJob, 
                            config,
                            job_configs,
                            [None] * len(job_configs),
                            True,
                            max_retries=1,
                            num_concurrent_jobs=-1)
    while not state.done():
        state.update_state()
        time.sleep(1)
    
    for result in state.results:
        print(result)
