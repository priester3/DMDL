from __future__ import absolute_import

import importlib

from .acceleration import configure_runtime, seed_everything


_TRAIN_BACKENDS = {
    "sysu": "dmdl_sysu",
    "regdb": "dmdl_regdb",
    "llcm": "dmdl_llcm",
}


def get_backend(dataset, stage):
    return _TRAIN_BACKENDS[dataset]


def run_legacy_training(cli_args, legacy_args):
    seed_everything(getattr(legacy_args, "seed", None))
    configure_runtime(legacy_args)
    module_name = get_backend(cli_args.dataset, cli_args.stage)
    module = importlib.import_module(module_name)
    return module.main(legacy_args)
