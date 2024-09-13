import time
import submitit_configs as sc
import submitit_tools as st
from dataclasses import dataclass, field
import subprocess
import os
from argparse import Namespace
names = ["exp1", "exp2", "exp3" ]

train_retrieval_tasks = [
    "put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]

retrieval_seeds = [
    1,
]

train_seeds = [
    2, 3,
]

@dataclass
class RobomimicTrainConfig(sc.BaseJobConfig):
    name: str = "dummy"
    retrieval_task: str  = "dummy"
    dataset_save_path: str = "dummy"
    retrieval_seed: int = -1
    train_seed: int = -1
    
job_configs = []

for name, retrieval_task in zip(names, train_retrieval_tasks):
    for ret_seed in retrieval_seeds:
        for tr_seed in train_seeds:  
            job_configs.append(
                RobomimicTrainConfig(name=name, retrieval_task=retrieval_task, dataset_save_path=os.path.join("retrieval", name, retrieval_task),
                                    checkpoint_path=os.path.join("retrieval", name, retrieval_task),
                                    retrieval_seed=ret_seed,
                                    train_seed=tr_seed)
                )
    
    
config = sc.SubmititExecutorConfig(root_folder="trash",
                                slurm_name="retrieval",
                                slurm_partition="gpu-a40",
                                timeout_min=60 * 24,
                                cpus_per_task=6,
                                mem_gb=128,
                                slurm_gpus_per_node="a40:1",
                                )



class TrainingJob(st.BaseJob):
    def __init__(self, job_config: RobomimicTrainConfig, wandb_config):
        super().__init__(job_config, wandb_config)
        self.job_config: RobomimicTrainConfig = job_config

    def _job_call(self):
        import robomimic
        from robomimic.scripts.train import main
        from robomimic.scripts.retrieval_utils import retrieve_data

        # do retrieval! 
        retrieved_data_save_path = retrieve_data(
            save_path=self.job_config.dataset_save_path,
            retrieval_task=self.job_config.retrieval_task,
            seed=25,
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
            k=1000   
        )
        cmd = f"python {os.path.dirname(robomimic.__file__)}/scripts/config_gen/bc_xfmr_gen.py --name {self.job_config.name} --train_task {retrieved_data_save_path} --env libero --no_pad --file --debug"
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
                filter_key=None,
                dataset_json=None,
                ckpt_path=None,
                output_dir=None,
                seed=None,
                debug=False,
            )
        main(args=args)
        return "done"
    
    def _initialize(self):
        pass



state = st.SubmititState(TrainingJob, 
                         config,
                         job_configs,
                         None,
                         True,
                         max_retries=1,
                         num_concurrent_jobs=-1)
while state.done() is False:
    state.update_state()
    time.sleep(1)
print('past state')           

# Process the results. The results are updated as jobs complete 
# so you can access this before
for result in state.results:
    print(result)
