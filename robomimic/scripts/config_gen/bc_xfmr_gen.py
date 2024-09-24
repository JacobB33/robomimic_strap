import robomimic
from robomimic.scripts.config_gen.helper import (
    get_generator,
    get_argparser,
    make_generator,
)
import os
# from robomimic.scripts.config_gen.simple_dataset_registry import *
from robomimic.macros import PERSON

def make_generator_helper(args):
    algo_name_short = "bc_xfmr"
    seq_length = 5


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
    
    config = dict(
        path=args.train_task,
        horizon=300,
        do_eval=True,
        filter_key=args.filter_key,
    )
    values_and_names = [([config], "human-50")]

    generator.add_param(key="experiment.name", name="", group=-1, values=[args.name])

    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=values_and_names,
    )
    if PERSON == "jacob":
        output_dir_path = f"/gscratch/weirdlab/jacob33/expdata/{args.env}/{args.mod}/{algo_name_short}"
    elif PERSON == "marius":
        output_dir_path = f"/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/students/mem1pi/robomimic_logs/{args.env}/{args.mod}/{algo_name_short}"
    else:
        assert False
        
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[output_dir_path],
    )

    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=-1,
        values=["all"],
    )

    ##### ALGORITHM #####

    if "train.batch_size" not in generator.parameters:
            generator.add_param(
                key="train.batch_size",
                name="",
                group=-1,
                values=[32],
            )
    if "train.num_epochs" not in generator.parameters:
        generator.add_param(
            key="train.num_epochs",
            name="",
            group=-1,
            values=[300],
        )

    generator.add_param(
        key="train.seq_length",
        name="",
        group=-1,
        values=[seq_length],
    )
    generator.add_param(
        key="train.frame_stack",
        name="",
        group=-1,
        values=[seq_length],
    )
    if args.no_pad:
        generator.add_param(
            key="train.pad_frame_stack",
            name="",
            group=-1,
            values=[False]
        )
        generator.add_param(
            key="train.pad_seq_length",
            name="",
            group=-1,
            values=[False]
        )
    generator.add_param(
        key="algo.transformer.context_length",
        name="",
        group=-1,
        values=[seq_length],
    )

    generator.add_param(
        key="algo.transformer.num_layers",
        name="",
        group=-1,
        values=[8],
    )
    generator.add_param(
        key="algo.transformer.embed_dim",
        name="",
        group=-1,
        values=[256],
    )
    generator.add_param(
        key="algo.transformer.num_heads",
        name="",
        group=-1,
        values=[4],
    )
    
    # proper language conditioned architecture
    generator.add_param(
            key="algo.language_conditioned",
            name="",
            group=-1,
            values=[True],
        )

    return generator


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
