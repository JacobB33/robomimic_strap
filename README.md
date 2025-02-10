# STRAP Policy Learning Repo
This repository contains the policy learning code based on [robomimic](https://github.com/ARISE-Initiative/robomimic/tree/robocasa) to reproduce the experiments in [STRAP](https://weirdlabuw.github.io/strap/).

-------
## Setup
Following the setup instructions in [STRAP retrieval code]([https://github.com/WEIRDLabUW/STRAP](https://github.com/WEIRDLabUW/STRAP?tab=readme-ov-file#setup)) to setup the base conda environment.

1. Active the conda environment:
    ```bash
    conda activate strap
    ```
2. Install the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/envs):
    ```bash
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO
    pip install -e .  ## NOTE: Do not install the requirements for libero
    ```
3. Install robomimic:
    ```bash
    git clone https://github.com/WEIRDLabUW/robomimic_strap.git
    cd robomimic_strap
    pip install -e .
    ```

You're all set!


-------
## Policy Learning
We extensively tested STRAP with BC-Transformer. To generate a configuration script, run
```
python robomimic/scripts/config_gen/bc_xfmr_gen.py --name <experiment-name> --train_ds_path <path to your train dataset>
```

The above generates a robomimic config file (json) and outputs a python command to start the training.

Here's an example:
```
python [PATH_TO_REPO]/robomimic_strap/robomimic/scripts/train.py --config [USER_DIR]/tmp/autogen_configs/ril/bc/libero/im/02-10-debug/02-10-25-08-12-19/json/seed_123_ds_human-50.json
```

Modify the config file (json) accordingly to adjust logging, training, and evaluation parameters. Add `--debug` to the config generation to generate lightweight runs for debugging.

-------
## Citation
```bibtex aiignore
@article{memmel2024strap,
  title={STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning},
  author={Memmel, Marius and Berg, Jacob and Chen, Bingqing and Gupta, Abhishek and Francis, Jonathan},
  journal={arXiv preprint arXiv:2412.15182},
  year={2024}
}
```

## Acknowledgments
This repository was modified from the official policy learning repo accompanying the [RoboCasa](https://robocasa.ai/) paper. 
