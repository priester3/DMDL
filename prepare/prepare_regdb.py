from common import QUERY_PER_ID, Sample, build_parser, copy_samples, ensure_dir, read_index_lines, resolve_roots


def parse_trials(raw_trials):
    trials = sorted(set(raw_trials))
    if not trials:
        raise ValueError("At least one trial must be provided.")
    return trials


def build_sample(data_root, line):
    relative_path, person_id = line.split()
    return Sample(src=data_root / relative_path, pid=person_id, cam="c1")


def load_samples(data_root, index_name):
    index_file = data_root / "idx" / index_name
    return [build_sample(data_root, line) for line in read_index_lines(index_file)]


def prepare_trial(data_root, output_root, trial):
    trial_counts = {}
    for output_name, train_index, test_index in (
        ("ir_modify", "train_thermal_{}.txt".format(trial), "test_thermal_{}.txt".format(trial)),
        ("rgb_modify", "train_visible_{}.txt".format(trial), "test_visible_{}.txt".format(trial)),
    ):
        target_root = ensure_dir(output_root / output_name / str(trial))
        test_samples = load_samples(data_root, test_index)
        train_samples = load_samples(data_root, train_index)
        trial_counts[output_name] = {
            "query": copy_samples(test_samples, target_root / "query", limit_per_pid=QUERY_PER_ID),
            "bounding_box_test": copy_samples(test_samples, target_root / "bounding_box_test"),
            "bounding_box_train": copy_samples(train_samples, target_root / "bounding_box_train"),
        }
    return trial_counts


def main():
    parser = build_parser("Prepare RegDB into the DMDL directory layout.")
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="RegDB trials to prepare.",
    )
    args = parser.parse_args()
    data_root, output_root = resolve_roots(args.data_root, args.output_root)

    all_counts = {}
    for trial in parse_trials(args.trials):
        all_counts[trial] = prepare_trial(data_root, output_root, trial)

    print("Prepared RegDB")
    print("  source: {}".format(data_root))
    print("  output: {}".format(output_root))
    print("  query_per_id: {}".format(QUERY_PER_ID))
    for trial in sorted(all_counts):
        print("  trial {}: {}".format(trial, all_counts[trial]))


if __name__ == "__main__":
    main()
