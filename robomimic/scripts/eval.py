"""
The main entry point for evaluating policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import os
import sys
import traceback

import torch
from copy import deepcopy

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.macros import LANG_KEY


def eval(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config, auto_remove_exp_dir=False, keep_exp_dir=True)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        print(dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get(LANG_KEY, None)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    eval_env_meta_list = []
    eval_shape_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
    for (dataset_i, dataset_cfg) in enumerate(config.train.data):
        do_eval = dataset_cfg.get("do_eval", True)
        if do_eval is not True:
            continue
        eval_env_meta_list.append(env_meta_list[dataset_i])
        eval_shape_meta_list.append(shape_meta_list[dataset_i])
        if ds_format == "libero":
            # env_meta_list[dataset_i]["env_name"] is same for all libero envs -> extract name from .bddl file
            eval_env_name_list.append(env_meta_list[dataset_i]["bddl_file_name"].split(".bddl")[0].split("/")[-1])
        else:
            eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
        horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)
    
    # create environments
    def env_iterator():
        for (env_meta, shape_meta, env_name) in zip(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list):
            def create_env_helper(env_i=0):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                    seed=config.train.seed * 1000 + env_i,
                )
                
                if config.train.data_format == "robosuite":
                    
                    # copy env_kwargs before modifying
                    env_kwargs_tmp = deepcopy(env_kwargs)
                    
                    # manually set env language
                    task_description = env_kwargs_tmp["env_meta"]["env_kwargs"]["env_"+LANG_KEY]
                    del env_kwargs_tmp["env_meta"]["env_kwargs"]["env_"+LANG_KEY]
                    
                    # only supports single robot: list -> string
                    env_kwargs_tmp["env_meta"]["env_kwargs"]["robots"] = env_kwargs_tmp["env_meta"]["env_kwargs"]["robots"][0]
                    # add robocasa wrapper to re-map obs keys
                    import robomimic.envs.env_base as EB
                    env_kwargs_tmp["env_meta"]["type"] = EB.EnvType.ROBOCASA_TYPE

                    # delete keys not supported by robosuite v1.4.1
                    del env_kwargs_tmp["seed"]
                    if hasattr(env_kwargs_tmp["env_meta"]["env_kwargs"], "generative_textures"):
                        del env_kwargs_tmp["env_meta"]["env_kwargs"]["generative_textures"]
                    
                    env = EnvUtils.create_env_from_metadata(**env_kwargs_tmp)
                    env.env_lang = task_description
                
                elif config.train.data_format == "libero":

                    from robomimic.envs.env_libero import EnvLibero
                    task_name = env_meta["env_name"]
                    task_description = env_meta["language_instruction"] if env_meta["env_"+LANG_KEY] is None else env_meta["env_"+LANG_KEY]
                    env = EnvLibero(env_name=task_name,
                                    env_meta=env_meta,
                                    render=False,
                                    render_offscreen=False, 
                                    use_image_obs=False,
                                    env_lang=task_description)
                
                else: # elif config.train.data_format == "robomimic":
                    env = EnvUtils.create_env_from_metadata(**env_kwargs)
                    # overwrite language with custom key
                    env.env_lang = env_meta["env_"+LANG_KEY]
                # handle environment wrappers
                env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable
                return env

            if config.experiment.rollout.batched:
                from tianshou.env import SubprocVectorEnv
                env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
                env = SubprocVectorEnv(env_fns)
                # env_name = env.get_env_attr(key="name", id=0)[0]
            else:
                env = create_env_helper()
                # env_name = env.name
            print(env)
            yield env

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )

    n_params = sum([sum(p.numel() for p in net.parameters()) for net in model.nets.values()])
    print(f"Initialized model with {n_params} parameters")
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    if config.algo.language_encoder == "clip":
        lang_encoder = LangUtils.CLIPLangEncoder(device=device)
    elif config.algo.language_encoder == "minilm":
        lang_encoder = LangUtils.MiniLMLangEncoder(device=device)
    
    # don't load training data in memory
    config.unlock()
    config.lock_keys()
    config.train.hdf5_cache_mode = None
    config.lock()
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=lang_encoder)
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # Evaluate the model by by running rollouts
    # do rollouts at fixed rate or if it's time to save a new ckpt
    
    # fetch model files in ckpt_path
    assert config.experiment.ckpt_path is not None
    ckpt_path = config.experiment.ckpt_path

    if type(ckpt_path) is list or (type(ckpt_path) is str and not ckpt_path.endswith(".pth")):
        file_names = os.listdir(ckpt_path)
        # only keep ckpt files
        file_names = [file_name for file_name in file_names if file_name.endswith(".pth")]
        # sort by epoch number
        file_names = sorted(file_names, key=lambda x: int(x.strip(".pth").split("_")[-1]))
    elif type(ckpt_path) is str:
        file_names = [ckpt_path]

    print("Found ckpt files: ", file_names)
    
    for file_name in file_names:

        # extract epoch number from file name
        epoch = int(file_name.strip(".pth").split("_")[-1])

        # load model weights
        ckpt_file_path = os.path.join(ckpt_path, file_name)
        print("LOADING MODEL WEIGHTS FROM " + ckpt_file_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_file_path)
        model.deserialize(ckpt_dict["model"])

        # wrap model as a RolloutPolicy to prepare for rollouts
        rollout_model = RolloutPolicy(
            model,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
            lang_encoder=lang_encoder,
        )

        num_episodes = config.experiment.rollout.n
        all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
            policy=rollout_model,
            envs=env_iterator(),
            horizon=eval_env_horizon_list,
            use_goals=config.use_goals,
            num_episodes=num_episodes,
            render=False,
            video_dir=video_dir if config.experiment.render_video else None,
            epoch=epoch,
            video_skip=config.experiment.get("video_skip", 5),
            terminate_on_success=config.experiment.rollout.terminate_on_success,
            del_envs_after_rollouts=True,
            data_logger=data_logger,
        )

    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.ckpt_path is not None:
        config.experiment.ckpt_path = args.ckpt_path

    if args.dataset is not None:
        # config.train.data = args.dataset

        config.train.data = [{
            "horizon": 700,
            "do_eval": True,
            "filter_key": None,
            "path": args.dataset
        }]

    if args.name is not None:
        config.experiment.name = args.name

    if args.seed is not None:
        config.train.seed = args.seed

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        eval(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Ckpt path, to override the one in the config
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="(optional) if provided, override the ckpt path defined in the config",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) if provided, override the seed defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()
    main(args)
