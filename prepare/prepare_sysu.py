from common import QUERY_PER_ID, Sample, build_parser, copy_samples, ensure_dir, resolve_roots


TEST_IDS = "test_id.txt"
TRAIN_IDS = "train_id.txt"
VAL_IDS = "val_id.txt"
IR_CAMERAS = ("cam3", "cam6")
RGB_CAMERAS = ("cam1", "cam2", "cam4", "cam5")


def read_ids(id_file):
    with id_file.open("r") as handle:
        values = handle.readline().strip().split(",")
    return ["{:04d}".format(int(value)) for value in values if value]


def iter_samples(data_root, cameras, person_ids):
    for person_id in sorted(person_ids):
        for camera in cameras:
            image_dir = data_root / camera / person_id
            if not image_dir.is_dir():
                continue
            for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
                yield Sample(src=image_path, pid=person_id, cam="c{}".format(camera[-1]))


def split_test_samples(data_root, cameras, person_ids):
    query_samples = []
    gallery_samples = []

    for person_id in sorted(person_ids):
        selected = 0
        for camera in cameras:
            image_dir = data_root / camera / person_id
            if not image_dir.is_dir():
                continue
            for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
                sample = Sample(src=image_path, pid=person_id, cam="c{}".format(camera[-1]))
                if selected < QUERY_PER_ID:
                    query_samples.append(sample)
                else:
                    gallery_samples.append(sample)
                selected += 1

    return query_samples, gallery_samples


def prepare_modality(data_root, output_root, output_name, cameras):
    target_root = ensure_dir(output_root / output_name)
    test_ids = read_ids(data_root / "exp" / TEST_IDS)
    train_ids = read_ids(data_root / "exp" / TRAIN_IDS) + read_ids(data_root / "exp" / VAL_IDS)

    query_samples, gallery_samples = split_test_samples(data_root=data_root, cameras=cameras, person_ids=test_ids)
    train_samples = list(iter_samples(data_root, cameras, train_ids))

    counts = {
        "query": copy_samples(query_samples, target_root / "query"),
        "bounding_box_test": copy_samples(gallery_samples, target_root / "bounding_box_test"),
        "bounding_box_train": copy_samples(train_samples, target_root / "bounding_box_train"),
    }
    return counts


def main():
    parser = build_parser("Prepare SYSU-MM01 into the DMDL directory layout.")
    args = parser.parse_args()
    data_root, output_root = resolve_roots(args.data_root, args.output_root)

    ir_counts = prepare_modality(data_root, output_root, "ir_modify", IR_CAMERAS)
    rgb_counts = prepare_modality(data_root, output_root, "rgb_modify", RGB_CAMERAS)

    print("Prepared SYSU-MM01")
    print("  source: {}".format(data_root))
    print("  output: {}".format(output_root))
    print("  query_per_id: {}".format(QUERY_PER_ID))
    print("  ir_modify: {}".format(ir_counts))
    print("  rgb_modify: {}".format(rgb_counts))


if __name__ == "__main__":
    main()
