from __future__ import print_function

import argparse
import os


ARCH_CHOICES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet_ibn50a",
    "resnet_ibn101a",
    "agw_one",
)


def resolve_checkpoint(path):
    if os.path.isdir(path):
        return os.path.join(path, "model_best.pth.tar")
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified DMDL evaluation entrypoint for SYSU, RegDB, and LLCM."
    )
    parser.add_argument("--dataset", choices=("sysu", "regdb", "llcm"), required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file or run directory.")
    parser.add_argument("-a", "--arch", type=str, default="agw_one", choices=ARCH_CHOICES)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling-type", type=str, default="gem")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--all-trials", action="store_true", help="Average RegDB over 10 trials.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    from clustercontrast.evaluators import run_evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()
