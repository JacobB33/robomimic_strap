import robomimic
from robomimic.scripts.config_gen.helper import (
    get_generator,
    get_argparser,
    make_generator,
)
from robomimic.scripts.config_gen.simple_dataset_registry import *


def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    robomimic_base_path = os.path.abspath(
        os.path.join(os.path.dirname(robomimic.__file__), os.pardir)
    )
    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(
            robomimic_base_path,
            "robomimic/exps/templates/bc_transformer.json",
        ),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    ##### DATA #####

    # Add training tasks to dataset
    values_and_names = [
        (
            get_ds_cfg(args.train_task, base_path=args.base_path, eval=args.eval_task),
            "human-50",
        )
    ]

    # Add evaluation tasks to dataset
    all_paths = [ds["path"] for ds in values_and_names[0][0]]
    for eval_task in args.eval_task:
        value = get_ds_cfg(
            eval_task,
            base_path=args.base_path,
            eval=args.eval_task,
        )[0]
        if value["path"] not in all_paths:
            values_and_names[0][0].append(value)

    generator.add_param(key="experiment.name", name="", group=-1, values=[args.name])

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

    ##### ALGORITHM #####

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

    # generator.add_param(
    #     key="algo.gmm.enabled",
    #     name="gmm",
    #     group=-1,
    #     values=[False],
    #     hidename=True,
    # )

    return generator


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
