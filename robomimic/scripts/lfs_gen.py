def generate_lfs_bsub_script(
        configs,
        time="16:00",
        cpus=8,
        mem=8000,
        gpus=1,
        queue="batch_v100",
        conda_env="libero_plain"
    ):
    
    assert len(configs) >= 2, "Need at least 2 configs to generate a .bsub script"

    bsub_template = f"""
#!/bin/bash

## Scheduler parameters ##
### https://inside-docupedia.bosch.com/confluence/display/CBDPS/%28dlp%29+%7C+Compute+Job+Management?src=contextnavpagetreemode ###

#BSUB -J multi_task[{1}-{len(configs)}]                                # job name | to run array of jobs define array of job names like: multi_task[1-5]
#BSUB -o /home/mem1pi/projects/lfs_logs/%J_%I.stdout    # optional: have output written to specific file
#BSUB -e /home/mem1pi/projects/lfs_logs/%J_%I.stderr    # optional: have errors written to specific file
#BSUB -W {time}                                         # fill in desired wallclock time [hours,]minutes (hours are optional)
#BSUB -n {cpus}                                         # min CPU cores,max CPU cores (max cores is optional)
#BSUB -M {mem}                                          # fill in required amount of RAM (in Mbyte) 125GB 128000 | 192GB 196608 | 400GB 409600
#BSUB -R "span[hosts=1]"                                # run on single host (if using more than 1 CPU cores)
#BSUB -gpu "num={gpus}:mode=exclusive_process:mps=no"   # exclusive gpu | 4 GPUs at a time total for interns
#BSUB -q {queue}                                        # optional: use highend nodes w/ Volta GPUs (default: Geforce GPUs)

## Job parameters ##

# activate conda
module unload conda1
module load conda/4.9.2
conda activate {conda_env}

## Mujoco headless rendering ##
export MUJOCO_GL=egl
    """

    # add 1st job
    bsub_template += f"""
if [ $(((LSB_JOBINDEX))) -eq 1 ]; then
    python /home/mem1pi/projects/robomimic_ret/robomimic/scripts/train.py --config {configs[0]}
    """

    # add 2nd - n-1th job
    for i, config in enumerate(configs[1:-1]):
        bsub_template += f"""
elif [ $(((LSB_JOBINDEX))) -eq {i+2} ]; then
    python /home/mem1pi/projects/robomimic_ret/robomimic/scripts/train.py --config {configs[i+1]}
    """
    
    # add nth job
    bsub_template += f"""
elif [ $(((LSB_JOBINDEX))) -eq {len(configs)} ]; then
    python /home/mem1pi/projects/robomimic_ret/robomimic/scripts/train.py --config {configs[-1]}
fi
    """

    # dump to .bsub file
    file_path = "job_script.bsub"
    with open(file_path, "w") as file:
        file.write(bsub_template)
    print(file_path)