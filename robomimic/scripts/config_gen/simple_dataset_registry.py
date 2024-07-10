from collections import OrderedDict
from copy import deepcopy
import os

SINGLE_STAGE_TASK_DATASETS = OrderedDict(
    PnPCounterToCab=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24",
    ),
    PnPCabToCounter=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24",
    ),
    PnPCounterToSink=dict(
        horizon=700,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25",
    ),
    PnPSinkToCounter=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2",
    ),
    PnPCounterToMicrowave=dict(
        horizon=600,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27",
    ),
    PnPMicrowaveToCounter=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26",
    ),
    PnPCounterToStove=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26",
    ),
    PnPStoveToCounter=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01",
    ),
    OpenSingleDoor=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24",
    ),
    CloseSingleDoor=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24",
    ),
    OpenDoubleDoor=dict(
        horizon=1000,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26",
    ),
    CloseDoubleDoor=dict(
        horizon=700,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29",
    ),
    OpenDrawer=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03",
    ),
    CloseDrawer=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30",
    ),
    TurnOnSinkFaucet=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25",
    ),
    TurnOffSinkFaucet=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25",
    ),
    TurnSinkSpout=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29",
    ),
    # exclude for now -> has navigation
    # TurnOnStove=dict(
    #     horizon=500,
    #     file_name="demo_gentex_im128_randcams.hdf5",
    # filter_key="human_50",
    # path="v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02",
    # ),
    TurnOffStove=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02",
    ),
    CoffeeSetupMug=dict(
        horizon=600,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25",
    ),
    CoffeeServeMug=dict(
        horizon=600,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01",
    ),
    CoffeePressButton=dict(
        horizon=300,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25",
    ),
    TurnOnMicrowave=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25",
    ),
    TurnOffMicrowave=dict(
        horizon=500,
        file_name="demo_gentex_im128_randcams.hdf5",
        filter_key="human_50",
        path="v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25",
    ),
    # exclude for now -> has navigation
    # NavigateKitchen=dict(
    #     horizon=500,
    #     file_name="demo_gentex_im128_randcams.hdf5",
    # filter_key="human_50",
    # path="v0.1/single_stage/kitchen_navigate/NavigateKitchen/2024-05-09",
    # ),
)

MULTI_STAGE_TASK_DATASETS = OrderedDict(
    ArrangeVegetables=dict(
        file_name="demo_gentex_im128.hdf5",
        filter_key="human_50",
        path="v0.1/multi_stage/chopping_food/ArrangeVegetables/2024-05-11",
        horizon=1200,
        activity="chopping_food",
    ),
    MicrowaveThawing=dict(
        file_name="demo_gentex_im128.hdf5",
        filter_key="human_50",
        path="v0.1/multi_stage/defrosting_food/MicrowaveThawing/2024-05-11",
        horizon=1000,
        activity="defrosting_food",
    ),
    RestockPantry=dict(
        file_name="demo_gentex_im128.hdf5",
        filter_key="human_50",
        path="v0.1/multi_stage/restocking_supplies/RestockPantry/2024-05-10",
        horizon=1000,
        activity="restocking_supplies",
    ),
    PreSoakPan=dict(
        file_name="demo_gentex_im128.hdf5",
        filter_key="human_50",
        path="v0.1/multi_stage/washing_dishes/PreSoakPan/2024-05-10",
        horizon=1500,
        activity="washing_dishes",
    ),
    PrepareCoffee=dict(
        file_name="demo_gentex_im128.hdf5",
        filter_key="human_50",
        path="v0.1/multi_stage/brewing/PrepareCoffee/2024-05-07",
        horizon=1000,
        activity="brewing",
    ),
)

VIOLA_REAL_TASK_DATASETS = OrderedDict(
    RealCapsuleCoffeeDomain=dict(
        file_name="robocasa_demo.hdf5",
        filter_key=None,
        path="viola/RealCapsuleCoffeeDomain_training_set",
        horizon=500,
        activity="coffee",
    ),
    RealKitchenBowlDomain=dict(
        file_name="robocasa_demo.hdf5",
        filter_key=None,
        path="viola/RealKitchenBowlDomain_training_set",
        horizon=500,
        activity="bowl",
    ),
    RealKitchenPlateForkDomain=dict(
        file_name="robocasa_demo.hdf5",
        filter_key=None,
        path="viola/RealKitchenPlateForkDomain_training_set",
        horizon=500,
        activity="plate_fork",
    ),
)

# TODO: add datasets here

ALL_DATASETS = OrderedDict()
ALL_DATASETS["single_stage"] = SINGLE_STAGE_TASK_DATASETS
ALL_DATASETS["multi_stage"] = MULTI_STAGE_TASK_DATASETS
ALL_DATASETS["viola_real"] = VIOLA_REAL_TASK_DATASETS

# TODO: add datasets here


def get_ds_cfg(
    ds_names,
    base_path,
    exclude_ds_names=None,
    overwrite_ds_lang=False,
    filter_key=None,
    eval=None,
    gen_tex=True,
    rand_cams=True,
):

    all_datasets = {}
    for k, v in ALL_DATASETS.items():
        all_datasets.update(v)

    # filter datsets by name or rule
    if ds_names == "all":
        ds_names = list(all_datasets.keys())
    if ds_names in ALL_DATASETS:
        ds_names = list(ALL_DATASETS[ds_names].keys())

    # TODO: add names or rules here

    elif ds_names == "pnp":
        ds_names = [name for name in all_datasets.keys() if "PnP" in name]
    elif isinstance(ds_names, str):
        ds_names = [ds_names]

    if exclude_ds_names is not None:
        ds_names = [name for name in ds_names if name not in exclude_ds_names]

    ret = []
    for name in ds_names:
        ds_meta = all_datasets[name]

        cfg = dict(horizon=ds_meta["horizon"])

        # determine whether eval on dataset
        if eval is None or name in eval:
            cfg["do_eval"] = True
        else:
            cfg["do_eval"] = False

        # if applicable overwrite the language stored in the dataset
        if overwrite_ds_lang is True:
            cfg["lang"] = ds_meta["lang"]

        # determine dataset path
        path_list = ds_meta.get("path", None)

        # skip if entry does not exist for this dataset src
        if path_list is None:
            continue

        if isinstance(path_list, str):
            path_list = [path_list]

        for path_i, path in enumerate(path_list):

            path = os.path.join(base_path, path)

            cfg_for_path = deepcopy(cfg)

            # determine dataset filter key
            if filter_key is not None:
                cfg_for_path["filter_key"] = filter_key
            else:
                cfg_for_path["filter_key"] = ds_meta[f"filter_key"]

            if "env_meta_update_dict" in ds_meta:
                cfg_for_path["env_meta_update_dict"] = ds_meta["env_meta_update_dict"]

            if not path.endswith(".hdf5"):
                # determine path
                if gen_tex is True and rand_cams is True:
                    path = os.path.join(path, "demo_gentex_im128_randcams.hdf5")
                elif gen_tex is True and rand_cams is False:
                    path = os.path.join(path, "demo_gentex_im128.hdf5")
                elif gen_tex is False and rand_cams is False:
                    path = os.path.join(path, "demo_im128.hdf5")
                else:
                    raise ValueError
            cfg_for_path["path"] = path

            if path_i > 0:
                cfg_for_path["do_eval"] = False

            ret.append(cfg_for_path)

    return ret
