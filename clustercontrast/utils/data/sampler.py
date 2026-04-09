from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import random
import torch
from torch.utils.data.sampler import Sampler


def _exclude_value_indices(items, value):
    return [index for index, item in enumerate(items) if item != value]


def _sample_group_indices(indexes, cams, anchor_index, anchor_cam, num_instances):
    selected = [anchor_index]
    needed = num_instances - 1
    if needed <= 0:
        return selected

    select_cams = _exclude_value_indices(cams, anchor_cam)
    if select_cams:
        replace = len(select_cams) < needed
        chosen = np.random.choice(select_cams, size=needed, replace=replace).tolist()
        selected.extend(indexes[idx] for idx in chosen)
        return selected

    select_indexes = _exclude_value_indices(indexes, anchor_index)
    if select_indexes:
        replace = len(select_indexes) < needed
        chosen = np.random.choice(select_indexes, size=needed, replace=replace).tolist()
        selected.extend(indexes[idx] for idx in chosen)
        return selected

    # Singleton cluster: repeat the anchor to preserve aligned inter-modality sampling.
    selected.extend([anchor_index] * needed)
    return selected


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, _, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, _, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = _exclude_value_indices(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = _exclude_value_indices(index, i)
                if not select_indexes:
                    continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)


class RandomMultipleGallerySamplerNoCam(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)

        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, _, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, _, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            index = self.pid_index[pid_i]

            select_indexes = _exclude_value_indices(index, i)
            if not select_indexes:
                continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)


class RandomMultipleGallerySamplerInterModality(Sampler):
    def __init__(self, data_source_rgb, data_source_ir, num_instances=4):
        super().__init__(data_source_rgb)
        super().__init__(data_source_ir)
        self.data_source_rgb = data_source_rgb
        self.index_pid_rgb = defaultdict(int)
        self.pid_cam_rgb = defaultdict(list)
        self.pid_index_rgb = defaultdict(list)
        self.data_source_ir = data_source_ir
        self.index_pid_ir = defaultdict(int)
        self.pid_cam_ir = defaultdict(list)
        self.pid_index_ir = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, _, cam) in enumerate(data_source_rgb):
            if pid < 0:
                continue
            self.index_pid_rgb[index] = pid
            self.pid_cam_rgb[pid].append(cam)
            self.pid_index_rgb[pid].append(index)

        self.pids_rgb = list(self.pid_index_rgb.keys())
        self.num_samples_rgb = len(self.pids_rgb)

        for index, (_, pid, _, cam) in enumerate(data_source_ir):
            if pid < 0:
                continue
            self.index_pid_ir[index] = pid
            self.pid_cam_ir[pid].append(cam)
            self.pid_index_ir[pid].append(index)

        self.pids_ir = list(self.pid_index_ir.keys())
        self.num_samples_ir = len(self.pids_ir)

        pids = list(set(self.pids_rgb+self.pids_ir))
        indices = torch.randperm(len(pids)).tolist()
        ret_rgb = []
        ret_ir = []

        for kid in indices:
            if pids[kid] not in self.pids_rgb:
                kid_rgb =  random.choice(self.pids_rgb)
            else:
                kid_rgb = pids[kid]
            
            if pids[kid] not in self.pids_ir:
                kid_ir = random.choice(self.pids_ir)
            else:
                kid_ir = pids[kid]
            
            i_rgb = random.choice(self.pid_index_rgb[kid_rgb])
            _, i_pid_rgb, _, i_cam_rgb = self.data_source_rgb[i_rgb]
            pid_i_rgb = self.index_pid_rgb[i_rgb]
            cams_rgb = self.pid_cam_rgb[pid_i_rgb]
            index_rgb = self.pid_index_rgb[pid_i_rgb]

            i_ir = random.choice(self.pid_index_ir[kid_ir])
            _, i_pid_ir, _, i_cam_ir = self.data_source_ir[i_ir]
            pid_i_ir = self.index_pid_ir[i_ir]
            cams_ir = self.pid_cam_ir[pid_i_ir]
            index_ir = self.pid_index_ir[pid_i_ir]

            ret_rgb.extend(_sample_group_indices(index_rgb, cams_rgb, i_rgb, i_cam_rgb, self.num_instances))
            ret_ir.extend(_sample_group_indices(index_ir, cams_ir, i_ir, i_cam_ir, self.num_instances))

        self.index_rgb = ret_rgb
        self.index_ir = ret_ir

    def __len__(self):
        return min(len(self.index_rgb), len(self.index_ir))

    def __iter__(self):
        return iter(list(range(len(self))))
