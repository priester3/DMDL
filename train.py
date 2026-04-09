from __future__ import print_function

from clustercontrast.dmdl.defaults import build_train_namespace
from clustercontrast.dmdl.legacy import run_legacy_training
from clustercontrast.dmdl.parser import build_unified_train_parser


def build_parser():
    return build_unified_train_parser()


def main():
    parser = build_parser()
    cli_args = parser.parse_args()
    legacy_args = build_train_namespace(cli_args)
    run_legacy_training(cli_args, legacy_args)


if __name__ == "__main__":
    main()
