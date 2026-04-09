from __future__ import absolute_import

import argparse
import os

from clustercontrast import models


LEGACY_DESCRIPTION = "Augmented Dual-Contrastive Aggregation Learning for USL-VI-ReID"
UNIFIED_DESCRIPTION = "Unified DMDL training entrypoint for SYSU, RegDB, and LLCM."
UNIFIED_ARCH_CHOICES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet_ibn50a",
    "resnet_ibn101a",
    "agw_one",
)


def str2bool(value):
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "y", "on"):
        return True
    if lowered in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got: {0}".format(value))


def build_unified_train_parser(root_dir=None):
    parser = argparse.ArgumentParser(description=UNIFIED_DESCRIPTION)
    parser.add_argument("--dataset", choices=("sysu", "regdb", "llcm"), required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--run-name", type=str, default=None, help="Current stage log directory name.")
    parser.add_argument("--stage1-name", type=str, default=None, help="Stage-1 log directory name used by stage 2.")
    parser.add_argument("--log-s2-name", type=str, default=None, help="Optional explicit stage-2 log directory name.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--logs-dir", type=str, default=None)
    parser.add_argument(
        "--root-dir",
        type=str,
        default=root_dir or os.getcwd(),
        help="Project root used to derive default data/log paths.",
    )

    parser.add_argument("-a", "--arch", type=str, default="agw_one", choices=UNIFIED_ARCH_CHOICES)
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=None)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--num-instances", type=int, default=16)
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--eval-step", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--pooling-type", type=str, default="gem")
    parser.add_argument("--miu", type=float, default=0.5)
    parser.add_argument("--lmada", type=float, default=1.0)
    parser.add_argument("--trial", type=int, default=1)

    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--eps-gap", type=float, default=0.02)
    parser.add_argument("--k1", type=int, default=None)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--topk", type=int, default=None)

    parser.add_argument("--use-hard", action="store_true")
    parser.add_argument("--no-cam", action="store_true")
    parser.add_argument("--runtime-mode", choices=("strict", "fast"), default=None)
    parser.add_argument("--amp-mode", choices=("no-amp", "fp16", "bf16"), default=None)
    parser.add_argument("--pin-memory", type=str2bool, default=None)
    parser.add_argument("--non-blocking", type=str2bool, default=None)
    parser.add_argument("--persistent-workers", type=str2bool, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--channels-last", type=str2bool, default=None)
    parser.add_argument("--compile", dest="use_compile", type=str2bool, default=None)
    parser.add_argument("--compile-mode", type=str, default=None)
    parser.add_argument("--compile-cache-dir", type=str, default=None)
    parser.add_argument("--fused-optimizer", dest="use_fused_optimizer", type=str2bool, default=None)
    return parser


def require_dataset_data_dir(parser, dataset_label):
    for action in parser._actions:
        if getattr(action, "dest", None) == "data_dir":
            action.default = None
            action.required = True
            action.help = "path to the {0} dataset root".format(dataset_label)
            return parser
    raise ValueError("Missing data_dir argument in parser for {0}.".format(dataset_label))


def build_legacy_stage_parser(
    dataset_name,
    working_dir,
    *,
    data_dir_default,
    logs_dir_default,
    epochs_default,
    log_s1_default,
    log_s2_default,
    batch_size_default=256,
    workers_default=8,
    height_default=288,
    width_default=144,
    num_instances_default=16,
    eps_default=0.6,
    eps_gap_default=0.02,
    k1_default=30,
    k2_default=6,
    arch_default="agw_one",
    features_default=0,
    dropout_default=0.0,
    momentum_default=0.2,
    lr_default=0.00035,
    weight_decay_default=5e-4,
    iters_default=200,
    step_size_default=20,
    seed_default=1,
    print_freq_default=10,
    eval_step_default=1,
    temp_default=0.05,
    pooling_type_default="gem",
    miu_default=0.5,
    include_topk=False,
    topk_default=10,
    include_lmada=False,
    lmada_default=1.0,
    include_trial=False,
    trial_default=1,
):
    parser = argparse.ArgumentParser(description=LEGACY_DESCRIPTION)
    parser.add_argument("-d", "--dataset", type=str, default=dataset_name)
    parser.add_argument("-b", "--batch-size", type=int, default=batch_size_default)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=workers_default)
    parser.add_argument("--height", type=int, default=height_default, help="input height")
    parser.add_argument("--width", type=int, default=width_default, help="input width")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=num_instances_default,
        help="each minibatch consist of (batch_size // num_instances) identities, and each identity has num_instances instances, default: 0 (NOT USE)",
    )

    parser.add_argument("--eps", type=float, default=eps_default, help="max neighbor distance for DBSCAN")
    parser.add_argument("--eps-gap", type=float, default=eps_gap_default, help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument("--k1", type=int, default=k1_default, help="hyperparameter for jaccard distance")
    parser.add_argument("--k2", type=int, default=k2_default, help="hyperparameter for jaccard distance")

    parser.add_argument("-a", "--arch", type=str, default=arch_default, choices=models.names())
    parser.add_argument("--features", type=int, default=features_default)
    parser.add_argument("--dropout", type=float, default=dropout_default)
    parser.add_argument("--momentum", type=float, default=momentum_default, help="update momentum for the hybrid memory")

    parser.add_argument("--lr", type=float, default=lr_default, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=weight_decay_default)
    parser.add_argument("--epochs", type=int, default=epochs_default)
    parser.add_argument("--iters", type=int, default=iters_default)
    parser.add_argument("--step-size", type=int, default=step_size_default)

    parser.add_argument("--seed", type=int, default=seed_default)
    parser.add_argument("--print-freq", type=int, default=print_freq_default)
    parser.add_argument("--eval-step", type=int, default=eval_step_default)
    parser.add_argument("--temp", type=float, default=temp_default, help="temperature for scaling contrastive loss")

    parser.add_argument("--data-dir", type=str, metavar="PATH", default=data_dir_default)
    parser.add_argument("--logs-dir", type=str, metavar="PATH", default=logs_dir_default or os.path.join(working_dir, "logs_{0}".format(dataset_name)))
    parser.add_argument("--pooling-type", type=str, default=pooling_type_default)
    parser.add_argument("--use-hard", action="store_true")
    parser.add_argument("--no-cam", action="store_true")

    parser.add_argument("--miu", type=float, default=miu_default)
    parser.add_argument("--log-s1-name", type=str, default=log_s1_default)
    parser.add_argument("--log-s2-name", type=str, default=log_s2_default)

    if include_trial:
        parser.add_argument("--trial", type=int, default=trial_default)
    if include_lmada:
        parser.add_argument("--lmada", type=float, default=lmada_default)
    if include_topk:
        parser.add_argument("--topk", type=int, default=topk_default)

    return parser


def detect_stage_from_cli(default_stage=1):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--stage", type=int, choices=(1, 2), default=default_stage)
    probe_args, _ = probe.parse_known_args()
    return probe_args.stage


def build_legacy_dataset_parser(
    stage,
    dataset_name,
    working_dir,
    *,
    common_defaults,
    stage_defaults,
):
    if stage not in stage_defaults:
        raise ValueError("Unsupported stage: {0}".format(stage))

    parser_kwargs = dict(common_defaults)
    parser_kwargs.update(stage_defaults[stage])
    parser = build_legacy_stage_parser(dataset_name, working_dir, **parser_kwargs)

    if not any(getattr(action, "dest", None) == "stage" for action in parser._actions):
        parser.add_argument("--stage", type=int, choices=(1, 2), default=stage)
    else:
        parser.set_defaults(stage=stage)

    return parser
