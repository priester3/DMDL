from __future__ import absolute_import

import os.path as osp


def resolve_dataset_dir(root, dataset_name, subdir):
    root = osp.abspath(root)
    candidates = []

    def add_candidate(path):
        normalized = osp.abspath(path)
        if normalized not in candidates:
            candidates.append(normalized)

    add_candidate(osp.join(root, subdir))
    add_candidate(osp.join(root, dataset_name, subdir))
    add_candidate(root)

    parent = osp.dirname(root)
    add_candidate(osp.join(parent, subdir))
    add_candidate(osp.join(parent, dataset_name, subdir))

    grandparent = osp.dirname(parent)
    add_candidate(osp.join(grandparent, dataset_name, subdir))

    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    return candidates[0]
