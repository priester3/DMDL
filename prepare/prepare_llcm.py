from common import QUERY_PER_ID, Sample, build_parser, copy_samples, ensure_dir, read_index_lines, resolve_roots


def build_sample(data_root, line):
    relative_path, person_id = line.split()
    image_path = data_root / relative_path
    camera_id = "c{}".format(image_path.name[7])
    return Sample(src=image_path, pid=person_id, cam=camera_id)


def load_samples(data_root, index_name):
    index_file = data_root / "idx" / index_name
    return [build_sample(data_root, line) for line in read_index_lines(index_file)]


def prepare_modality(data_root, output_root, output_name, train_index, test_index):
    target_root = ensure_dir(output_root / output_name)
    test_samples = load_samples(data_root, test_index)
    train_samples = load_samples(data_root, train_index)

    counts = {
        "query": copy_samples(test_samples, target_root / "query", limit_per_pid=QUERY_PER_ID),
        "bounding_box_test": copy_samples(test_samples, target_root / "bounding_box_test"),
        "bounding_box_train": copy_samples(train_samples, target_root / "bounding_box_train"),
    }
    return counts


def main():
    parser = build_parser("Prepare LLCM into the DMDL directory layout.")
    args = parser.parse_args()
    data_root, output_root = resolve_roots(args.data_root, args.output_root)

    ir_counts = prepare_modality(
        data_root=data_root,
        output_root=output_root,
        output_name="ir_modify",
        train_index="train_nir.txt",
        test_index="test_nir.txt",
    )
    rgb_counts = prepare_modality(
        data_root=data_root,
        output_root=output_root,
        output_name="rgb_modify",
        train_index="train_vis.txt",
        test_index="test_vis.txt",
    )

    print("Prepared LLCM")
    print("  source: {}".format(data_root))
    print("  output: {}".format(output_root))
    print("  query_per_id: {}".format(QUERY_PER_ID))
    print("  ir_modify: {}".format(ir_counts))
    print("  rgb_modify: {}".format(rgb_counts))


if __name__ == "__main__":
    main()
