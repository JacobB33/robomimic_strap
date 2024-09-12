import time
import submitit_configs as sc
import submitit_tools as st
from dataclasses import dataclass, field
from robomimic.scripts.train import main
import subprocess
import os
from argparse import Namespace
import robomimic
names = ["exp1",
         "exp2"]

train_tasks = [
    "pick_the_book_1k",
    "pick_the_book_1k",
]

@dataclass
class RobomimicTrainConfig(sc.BaseJobConfig):
    arguments: Namespace = None

class TrainingJob(st.BaseJob):
    def __init__(self, job_config: RobomimicTrainConfig, wandb_config):
        super().__init__(job_config, wandb_config)
        self.job_config: RobomimicTrainConfig = job_config

    def _job_call(self):
        main(args=self.job_config.arguments)
        print("done")
        return "done"
    
    def _initialize(self):
        pass
    
job_configs = []
for name, train_task in zip(names, train_tasks):
    cmd = f"python {os.path.dirname(robomimic.__file__)}/scripts/config_gen/bc_xfmr_gen.py --name {name} --train_task {train_task} --env libero --no_pad"
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
                debug=False
            )
            job_configs.append(RobomimicTrainConfig(arguments=args))

  
print("creating config ")
config = sc.SubmititExecutorConfig(root_folder="trash",
                                slurm_name="retrieval",
                                slurm_partition="gpu-a40",
                                timeout_min=60 * 24,
                                cpus_per_task=6,
                                mem_gb=128,
                                slurm_gpus_per_node="a40:1",
                                )
state = st.SubmititState(TrainingJob, 
                         config,
                         job_configs,
                         None,
                         True,
                         max_retries=2,
                         num_concurrent_jobs=-1)
while state.done() is False:
    state.update_state()
    time.sleep(1)
print('past state')           

# Process the results. The results are updated as jobs complete 
# so you can access this before
for result in state.results:
    print(result)
