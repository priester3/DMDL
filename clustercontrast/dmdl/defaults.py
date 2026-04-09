from __future__ import absolute_import

import os
from argparse import Namespace


DATASETS = ("sysu", "regdb", "llcm")
STAGES = (1, 2)


_DATASET_COMMON_DEFAULTS = {
    "sysu": {
        "workers": 8,
        "k1": 30,
        "seed": 1,
        "runtime_mode": "fast",
        "amp_mode": "fp16",
    },
    "regdb": {
        "workers": 8,
        "k1": 30,
        "seed": 1,
        "runtime_mode": "fast",
        "amp_mode": "fp16",
    },
    "llcm": {
        "workers": 8,
        "k1": 25,
        "seed": 1,
        "runtime_mode": "fast",
        "amp_mode": "fp16",
    },
}


_STAGE_DEFAULTS = {
    ("sysu", 1): {
        "batch_size": 256,
        "epochs": 50,
        "iters": 200,
        "step_size": 20,
        "topk": 3,
        "log_name": "sysu_s1/dmdl",
    },
    ("sysu", 2): {
        "batch_size": 256,
        "epochs": 50,
        "iters": 200,
        "step_size": 20,
        "topk": 10,
        "log_name": "sysu_s2/dmdl",
    },
    ("regdb", 1): {
        "batch_size": 256,
        "epochs": 30,
        "iters": 100,
        "step_size": 20,
        "topk": 3,
        "log_name": "regdb_s1/dmdl",
    },
    ("regdb", 2): {
        "batch_size": 256,
        "epochs": 30,
        "iters": 100,
        "step_size": 20,
        "topk": 10,
        "log_name": "regdb_s2/dmdl",
    },
    ("llcm", 1): {
        "batch_size": 256,
        "epochs": 50,
        "iters": 200,
        "step_size": 20,
        "topk": 3,
        "log_name": "llcm_s1/dmdl",
    },
    ("llcm", 2): {
        "batch_size": 256,
        "epochs": 50,
        "iters": 200,
        "step_size": 20,
        "topk": 10,
        "log_name": "llcm_s2/dmdl",
    },
}


def default_logs_dir(dataset, root_dir):
    return os.path.join(root_dir, "logs_{0}".format(dataset))


def default_data_dir(dataset, root_dir):
    project_root = os.path.abspath(os.path.join(root_dir, os.pardir))
    if dataset == "llcm":
        return os.path.join(project_root, "dataset", "LLCM")
    return os.path.join(project_root, "dataset", dataset)


def _dataset_argument(dataset):
    if dataset == "sysu":
        return "sysu_all"
    return dataset


def build_train_namespace(args):
    dataset = args.dataset
    stage = args.stage
    common_defaults = _DATASET_COMMON_DEFAULTS[dataset]
    stage_defaults = _STAGE_DEFAULTS[(dataset, stage)]
    root_dir = args.root_dir
    resolved_workers = args.workers if args.workers is not None else common_defaults["workers"]
    resolved_k1 = args.k1 if args.k1 is not None else common_defaults["k1"]

    log_name = args.run_name or stage_defaults["log_name"]
    stage1_name = args.stage1_name or _STAGE_DEFAULTS[(dataset, 1)]["log_name"]

    return Namespace(
        stage=stage,
        dataset=_dataset_argument(dataset),
        batch_size=args.batch_size if args.batch_size is not None else stage_defaults["batch_size"],
        test_batch=args.test_batch,
        workers=resolved_workers,
        height=args.height,
        width=args.width,
        num_instances=args.num_instances,
        eps=args.eps,
        eps_gap=args.eps_gap,
        k1=resolved_k1,
        k2=args.k2,
        arch=args.arch,
        features=args.features,
        dropout=args.dropout,
        momentum=args.momentum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs if args.epochs is not None else stage_defaults["epochs"],
        iters=args.iters if args.iters is not None else stage_defaults["iters"],
        step_size=args.step_size if args.step_size is not None else stage_defaults["step_size"],
        seed=args.seed if args.seed is not None else common_defaults["seed"],
        print_freq=args.print_freq,
        eval_step=args.eval_step,
        temp=args.temp,
        data_dir=args.data_dir or default_data_dir(dataset, root_dir),
        logs_dir=args.logs_dir or default_logs_dir(dataset, root_dir),
        pooling_type=args.pooling_type,
        use_hard=args.use_hard,
        no_cam=args.no_cam,
        miu=args.miu,
        log_s1_name=log_name if stage == 1 else stage1_name,
        log_s2_name=log_name if stage == 2 else args.log_s2_name or _STAGE_DEFAULTS[(dataset, 2)]["log_name"],
        lmada=args.lmada,
        topk=args.topk if args.topk is not None else stage_defaults["topk"],
        trial=args.trial,
        runtime_mode=common_defaults["runtime_mode"] if getattr(args, "runtime_mode", None) is None else args.runtime_mode,
        amp_mode=common_defaults["amp_mode"] if getattr(args, "amp_mode", None) is None else args.amp_mode,
        pin_memory=True if getattr(args, "pin_memory", None) is None else args.pin_memory,
        non_blocking=True if getattr(args, "non_blocking", None) is None else args.non_blocking,
        persistent_workers=(resolved_workers > 0) if getattr(args, "persistent_workers", None) is None else args.persistent_workers,
        prefetch_factor=2 if getattr(args, "prefetch_factor", None) is None else args.prefetch_factor,
        channels_last=False if getattr(args, "channels_last", None) is None else args.channels_last,
        use_compile=False if getattr(args, "use_compile", None) is None else args.use_compile,
        compile_mode="default" if getattr(args, "compile_mode", None) is None else args.compile_mode,
        compile_cache_dir=getattr(args, "compile_cache_dir", None) or os.path.join(root_dir, ".cache", "torch_compile"),
        use_fused_optimizer=False if getattr(args, "use_fused_optimizer", None) is None else args.use_fused_optimizer,
    )
