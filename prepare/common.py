import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2


QUERY_PER_ID = 4


@dataclass(frozen=True)
class Sample:
    src: Path
    pid: str
    cam: str

    @property
    def output_name(self):
        return "{}_{}_{}".format(self.pid, self.cam, self.src.name)


def build_parser(description):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Path to the original downloaded dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Path where ir_modify/rgb_modify will be created. Defaults to --data-root.",
    )
    return parser


def resolve_roots(data_root, output_root=None):
    data_root = data_root.expanduser()
    output_root = (output_root or data_root).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError("Dataset root does not exist: {}".format(data_root))
    return data_root, output_root


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_index_lines(index_file):
    if not index_file.is_file():
        raise FileNotFoundError("Index file does not exist: {}".format(index_file))
    with index_file.open("r") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def copy_samples(samples, dst_dir, limit_per_pid=None):
    ensure_dir(dst_dir)
    copied = 0
    pid_counts = defaultdict(int)

    for sample in samples:
        if limit_per_pid is not None:
            pid_counts[sample.pid] += 1
            if pid_counts[sample.pid] > limit_per_pid:
                continue
        copy2(str(sample.src), str(dst_dir / sample.output_name))
        copied += 1

    return copied
