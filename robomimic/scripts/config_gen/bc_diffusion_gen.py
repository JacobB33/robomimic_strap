import robomimic
from robomimic.scripts.config_gen.helper import (
    get_generator,
    get_argparser,
    make_generator,
)
from robomimic.scripts.config_gen.simple_dataset_registry import *


def make_generator_helper(args):
    algo_name_short = "diffusion_policy"

    robomimic_base_path = os.path.abspath(
        os.path.join(os.path.dirname(robomimic.__file__), os.pardir)
    )
    generator = get_generator(
        algo_name="diffusion_policy",
        config_file=os.path.join(
            robomimic_base_path,
            "robomimic/exps/templates/diff_base.json" if args.co_train else "robomimic/exps/templates/diff_base_co.json",
        ),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    ##### DATA #####
    # Add training tasks to dataset
    if args.co_train:
        config0 = dict(
                path=args.train_task,
                horizon=300,
                do_eval=True,
                filter_key="demos",
                weight=1.0
            )
        config1 = dict(
                path=args.train_task,
                horizon=300,
                do_eval=False,
                filter_key="retrieval",
                weight=1.0
            )
        values_and_names = [([config0, config1], "co_train")]
        generator.add_param(key="train.normalize_weights_by_ds_size", name="", group=-1, values=[True])
        generator.add_param(
            key="train.hdf5_cache_mode",
            name="",
            group=-1,
            values=[None],
        )
    else:
        config = dict(
            path=args.train_task,
            horizon=300,
            do_eval=True,
            filter_key=args.filter_key,
        )
        values_and_names = [([config], "mix_train")]
        generator.add_param(
            key="train.hdf5_cache_mode",
            name="",
            group=-1,
            values=["all"],
        )
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
            "/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/students/mem1pi/robomimic_logs/{env}/{mod}/{algo_name_short}".format(
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
