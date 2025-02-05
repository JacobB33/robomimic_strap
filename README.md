# Strap Policy Learning Repo
This was built on robomimic with some modifications to work on the LIBERO environments


## Installation
Create a conda environment:
```
conda create -n strap python=3.10
conda activate strap
```

Insall LIBERO and STRAP. 
To install LIBERO run:
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .  ## NOTE: Do not install the requirements for libero
```

To install strap run:
```
git clone https://github.com/WEIRDLabUW/STRAP.git
cd STRAP
pip install -e .
```

Then install this repository
```
cd robomimic_strap
pip install -e .
```

-------
## Training
There are a number of algorithms to choose from. We offer official support for BC-Transformer. Users can also adapt the code to run Diffusion Policy, ACT, etc.

Each algorithm has its own config generator script. For example for BC-Transformer policy run:
```
python robomimic/scripts/config_gen/bc_xfmr_gen.py --name <experiment-name> --train_ds_path <path to your train dataset>
```
Modify this file accordingly, depending on which datasets you are training on and whether you are running evaluations.

Note: You can add `--debug` to generate small runs for testing.

After running this script, you just need to run the command(s) that are outputted.

### Acknowledgments:
This repository was modified from the official policy learning repo accompanying the [RoboCasa](https://robocasa.ai/) paper. 


