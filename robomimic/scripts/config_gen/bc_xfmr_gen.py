from robomimic.scripts.config_gen.helper import *


def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(
            base_path, "robomimic/exps/templates/bc_transformer.json"
            # base_path, "robomimic/exps/templates/bc_transformer_film.json"
        ),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    ### Multi-task training on atomic tasks ###
    EVAL_TASKS = args.eval_task
    # [
    #     "TurnOnMicrowave",
    # ]  # or evaluate all tasks by setting EVAL_TASKS = None
    
    values_and_names = None

    # Get evaluation tasks and add them to dataset
    eval_tasks = []
    for st in EVAL_TASKS:
        eval_tasks.append(get_ds_cfg(
                        st,
                        src="human",
                        eval=EVAL_TASKS,
                        filter_key="50_demos",
                    )[0])
        
    # Single task aka eval tasks
    if args.train_task == "single":
        values_and_names = [
            (
                eval_tasks,
                "human-50",
            )
        ]
    elif args.train_task == "pnp":
        values_and_names = [
            (
                get_ds_cfg("pnp", src="human", eval=EVAL_TASKS, filter_key="50_demos"),
                "human-50",
            )
        ]
    elif args.train_task == "pnp_half":
        values_and_names = [
            (
                get_ds_cfg("pnp_half", src="human", eval=EVAL_TASKS, filter_key="50_demos"),
                "human-50",
            )
        ]
    elif args.train_task == "turn":
        values_and_names = [
            (
                get_ds_cfg("turn", src="human", eval=EVAL_TASKS, filter_key="50_demos"),
                "human-50",
            )
        ]
    elif args.train_task == "openclose":
        values_and_names = [
            (
                get_ds_cfg("openclose", src="human", eval=EVAL_TASKS, filter_key="50_demos"),
                "human-50",
            )
        ]
    elif args.train_task == "openclose_half":
        values_and_names = [
            (
                get_ds_cfg("openclose_half", src="human", eval=EVAL_TASKS, filter_key="50_demos"),
                "human-50",
            )
        ]
    elif args.train_task == "all":
        values_and_names = [
            (
                get_ds_cfg(
                    "single_stage", src="human", eval=EVAL_TASKS, filter_key="50_demos"
                ),
                "human-50",
            )
        ]
    elif args.train_task == "viola_real":
         values_and_names = [
            (
                get_ds_cfg(
                "viola_real", src="human", eval=EVAL_TASKS, filter_key=None
            ),
            "viola-50"
            )
        ]

    else:
        ValueError("Invalid task")
    # TODO add dataset combination support in get_ds_cfg

    # Add evaluation tasks to dataset
    all_paths = [ds["path"] for ds in values_and_names[0][0]]
    for eval_task in eval_tasks:
        if eval_task["path"] not in all_paths:
            values_and_names[0][0].append(eval_task)

    # get_ds_cfg(
    #     "viola_real", src="human", eval=EVAL_TASKS, filter_key=None
    # ),
    # "viola-50"
    
    generator.add_param(
        key="experiment.name",
        name="",
        group=-1,
        values=[args.name]
            )

    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=values_and_names,
    )

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    # generator.add_param(
    #     key="observation.encoder.rgb.core_kwargs.backbone_class",
    #     name="backbone",
    #     group=1234,
    #     values=[
    #         "ResNet18ConvFiLM",
    #         # "ResNet50Conv",
    #     ],
    # )
    # generator.add_param(
    #     key="observation.encoder.rgb.core_kwargs.feature_dimension",
    #     name="visdim",
    #     group=1234,
    #     values=[
    #         64,
    #         # 512,
    #     ],
    # )

    # # pass language to transformer
    # generator.add_param(
    #     key="algo.language_conditioned",
    #     name="",
    #     group=1234,
    #     values=[
    #         # True,
    #         False,
    #     ],
    #     hidename=True,
    # )

    # # change default settings: predict 10 steps into future
    # generator.add_param(
    #     key="algo.transformer.pred_future_acs",
    #     name="predfuture",
    #     group=1,
    #     values=[
    #         True,
    #         # False,
    #     ],
    #     hidename=True,
    # )
    # generator.add_param(
    #     key="algo.transformer.supervise_all_steps",
    #     name="supallsteps",
    #     group=1,
    #     values=[
    #         True,
    #         # False,
    #     ],
    #     hidename=True,
    # )
    # generator.add_param(
    #     key="algo.transformer.causal",
    #     name="causal",
    #     group=1,
    #     values=[
    #         False,
    #         # True,
    #     ],
    #     hidename=True,
    # )
    # generator.add_param(
    #     key="train.seq_length",
    #     name="",
    #     group=-1,
    #     values=[10],
    #     hidename=True,
    # )

    # # don't use GMM
    # generator.add_param(
    #     key="algo.gmm.enabled",
    #     name="gmm",
    #     group=-1,
    #     values=[False],
    #     hidename=True,
    # )

    # generator.add_param(
    #     key="algo.gmm.min_std",
    #     name="mindstd",
    #     group=271314,
    #     values=[
    #         0.03,
    #         # 0.0001,
    #     ],
    #     hidename=True,
    # )
    # generator.add_param(
    #     key="train.max_grad_norm",
    #     name="maxgradnorm",
    #     group=18371,
    #     values=[
    #         # None,
    #         100.0,
    #     ],
    #     hidename=True,
    # )

    # automatic precision
    generator.add_param(
        key="algo.amp",
        name="",
        group=1234,
        values=[
            # True,
            False,
        ],
        hidename=True,
    )

    return generator


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
