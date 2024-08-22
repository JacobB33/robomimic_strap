"""
The main entry point for training policies.

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
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

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


def train(config, device):
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
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config, timestamp=False, auto_remove_exp_dir=True)
    # log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

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
        env_meta["env_lang"] = dataset_cfg.get("lang", None)
        if ds_format == "libero":
            env_meta["env_lang"] = env_meta["language_instruction"]
            
            # print("Language Instruction: {}".format(env_meta["env_lang"]))

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        #TODO: This env_meta_list seems to not be used
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
                env = EnvUtils.create_env_from_metadata(**env_kwargs)

                # TODO if only single demo, load env model file and ep_meta
                # dataset_path = "/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/students/mem1pi/datasets/retrieval/sub_traj_len_25_stride_20/TurnOnMicrowave"
                # demo_key = "demo_14"
                # # import h5py
                # # with h5py.File(dataset_path, "r") as f:

                # # ep_meta_tmp = json.loads(f["data"][demo_key].attrs["ep_meta"])
                # ep_meta_tmp = json.load(open(os.path.join(dataset_path, "ep_meta.json"), "r"))
                
                # env.env.set_ep_meta(ep_meta_tmp)
                # env.env.reset()

                # # model_file = f["data"][demo_key].attrs["model_file"]
                # model_file = open(os.path.join(dataset_path, "model_file.mjcf"), "r").read()
                # xml = env.env.edit_model_xml(model_file)
                # env.env.reset_from_xml_string(xml)
                # env.env.sim.reset()

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

    # if checkpoint is specified, load in model weights
    ckpt_path = config.experiment.ckpt_path
    if ckpt_path is not None:
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    if config.algo.language_encoder == "clip":
        lang_encoder = LangUtils.CLIPLangEncoder(device=device)
    elif config.algo.language_encoder == "minilm":
        lang_encoder = LangUtils.MiniLMLangEncoder(device=device)
        
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

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
        # modified,
        pin_memory=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in eval_env_name_list} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in eval_env_name_list} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(0, config.train.num_epochs + 1): # epoch numbers start at 1
        if epoch > 0:
            step_log = TrainUtils.run_epoch(
                model=model,
                data_loader=train_loader,
                epoch=epoch,
                num_steps=train_num_steps,
                obs_normalization_stats=obs_normalization_stats,
            )
            model.on_epoch_end(epoch)

            # setup checkpoint path
            epoch_ckpt_name = "model_epoch_{}".format(epoch)

            # check for recurring checkpoint saving conditions
            should_save_ckpt = False
            if config.experiment.save.enabled:
                time_check = (config.experiment.save.every_n_seconds is not None) and \
                    (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
                epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                    (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
                epoch_list_check = (epoch in config.experiment.save.epochs)
                should_save_ckpt = (time_check or epoch_check or epoch_list_check)
            ckpt_reason = None
            if should_save_ckpt:
                last_ckpt_time = time.time()
                ckpt_reason = "time"

            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Train/{}".format(k), v, epoch)

            # Evaluate the model on validation set
            if config.experiment.validate:
                with torch.no_grad():
                    step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
                for k, v in step_log.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                    else:
                        data_logger.record("Valid/{}".format(k), v, epoch)

                print("Validation Epoch {}".format(epoch))
                print(json.dumps(step_log, sort_keys=True, indent=4))

                # save checkpoint if achieve new best validation loss
                valid_check = "Loss" in step_log
                if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                    best_valid_loss = step_log["Loss"]
                    if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                        epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                        should_save_ckpt = True
                        ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason
        else:
            should_save_ckpt = False
            epoch_ckpt_name = "model_epoch_{}".format(epoch)
            ckpt_reason = None

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) #or (should_save_ckpt and ckpt_reason == "time") # remove this section condition, not desired when rollouts are expensive and saving frequent checkpoints
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
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

            #### move this code to rollout_with_stats function to log results one by one ####
            # # summarize results from rollouts to tensorboard and terminal
            # for env_name in all_rollout_logs:
            #     rollout_logs = all_rollout_logs[env_name]
            #     for k, v in rollout_logs.items():
            #         if k.startswith("Time_"):
            #             data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
            #         else:
            #             data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

            #     print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
            #     print('Env: {}'.format(env_name))
            #     print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # check if we need to save model MSE
        should_save_mse = False
        if config.experiment.mse.enabled:
            if config.experiment.mse.every_n_epochs is not None and epoch % config.experiment.mse.every_n_epochs == 0:
                should_save_mse = True
            if config.experiment.mse.on_save_ckpt and should_save_ckpt:
                should_save_mse = True
        if should_save_mse:
            print("Computing MSE ...")
            if config.experiment.mse.visualize:
                save_vis_dir = os.path.join(vis_dir, epoch_ckpt_name)
            else:
                save_vis_dir = None
            mse_log, vis_log = model.compute_mse_visualize(
                trainset,
                validset,
                num_samples=config.experiment.mse.num_samples,
                savedir=save_vis_dir,
            )    
            for k, v in mse_log.items():
                data_logger.record("{}".format(k), v, epoch)
            
            for k, v in vis_log.items():
                data_logger.record("{}".format(k), v, epoch, data_type='image')


            print("MSE Log Epoch {}".format(epoch))
            print(json.dumps(mse_log, sort_keys=True, indent=4))
        
        # # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        # should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        # if video_paths is not None and not should_save_video:
        #     for env_name in video_paths:
        #         os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:    
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

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

    # print(config.train.action_keys)
    # exit()
    if args.ckpt_path is not None:
        config.experiment.ckpt_path = args.ckpt_path

    if args.dataset_json is not None:
        config.train.data = [json.loads(args.dataset_json)]
        
    if args.dataset is not None:
        assert len(config.train.data) == 1, "dataset arg overwrite only supported for single dataset"
        
        # config.train.data = args.dataset
        filter_key = config.train.data[0]["filter_key"] if args.filter_key is None else args.filter_key
        do_eval = config.train.data[0]["do_eval"]
        horizon = config.train.data[0]["horizon"]
        
        config.train.data = [{
            "horizon": horizon,
            "do_eval": do_eval,
            "filter_key": filter_key,
            "path": args.dataset
        }]

    if args.name is not None:
        config.experiment.name = args.name

    if args.seed is not None:
        config.train.seed = args.seed

    config.experiment.name = config.experiment.name # + "-" + str(config.train.seed)
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    config.train.output_dir = os.path.join(config.train.output_dir, str(config.train.seed))

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
        train(config, device=device)
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
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) if provided, override the filter_key defined in the config",
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset list defined in the config",
    )

    # Ckpt path, to override the one in the config
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="(optional) if provided, override the ckpt path defined in the config",
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="(optional) if provided, override the output directory defined in the config",
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
