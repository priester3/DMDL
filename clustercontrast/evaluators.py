from __future__ import absolute_import, print_function

import os
import random
import time
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch import nn
from torchvision.transforms import InterpolationMode

from clustercontrast import models
from clustercontrast.utils import to_torch
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.meters import AverageMeter
from clustercontrast.utils.rerank import re_ranking
from clustercontrast.utils.serialization import load_checkpoint
from .dmdl.acceleration import get_inference_autocast_context

from .evaluation_metrics import cmc, mean_ap


def fliplr(img):
    return torch.flip(img, dims=[3])


def extract_cnn_feature(model, inputs, mode):
    inputs = to_torch(inputs).cuda(non_blocking=True)
    outputs = model(inputs, inputs, modal=mode)
    return outputs.data.cpu()


def extract_features(model, data_loader, print_freq=50, flip=True, mode=0, runtime_args=None):
    del flip
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    camids = OrderedDict()

    end = time.time()
    with torch.inference_mode():
        for i, (imgs, fnames, pids, cids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            imgs = to_torch(imgs).cuda(non_blocking=True)
            flipped = fliplr(imgs)
            with get_inference_autocast_context(runtime_args):
                outputs = model(imgs, imgs, modal=mode).detach().float().cpu()
                outputs_flip = model(flipped, flipped, modal=mode).detach().float().cpu()

            for fname, output, output_flip, pid, cid in zip(fnames, outputs, outputs_flip, pids, cids):
                features[fname] = (output.detach() + output_flip.detach()) / 2.0
                labels[fname] = pid
                camids[fname] = cid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(
                    "Extract Features: [{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\t".format(
                        i + 1,
                        len(data_loader),
                        batch_time.val,
                        batch_time.avg,
                        data_time.val,
                        data_time.avg,
                    )
                )

    return features, labels, camids


def extract_features_tensor(model, data_loader, print_freq=50, mode=0, normalize=False, runtime_args=None, keep_on_gpu=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = []
    camids = []

    end = time.time()
    with torch.inference_mode():
        for i, (imgs, _, _, cids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            imgs = to_torch(imgs).cuda(non_blocking=True)
            flipped = fliplr(imgs)
            with get_inference_autocast_context(runtime_args):
                merged = (model(imgs, imgs, modal=mode) + model(flipped, flipped, modal=mode)) / 2.0
            merged = merged.float()
            if normalize:
                merged = torch.nn.functional.normalize(merged, dim=1)
            features.append(merged if keep_on_gpu else merged.cpu())
            camids.append(cids.clone())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(
                    "Extract Features: [{}/{}]\tTime {:.3f} ({:.3f})\tData {:.3f} ({:.3f})\t".format(
                        i + 1,
                        len(data_loader),
                        batch_time.val,
                        batch_time.avg,
                        data_time.val,
                        data_time.avg,
                    )
                )

    if not features:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(features, dim=0), torch.cat(camids, dim=0)


def extract_cnn_feature_dis(model, masker, inputs, mode, epoch):
    inputs = to_torch(inputs).cuda(non_blocking=True)
    outputs = model(inputs, inputs, modal=mode)
    masker_sup = masker(outputs.detach())
    if epoch < 5:
        masker_sup = torch.ones_like(outputs.detach())
    outputs = outputs * masker_sup
    return outputs.data.cpu()


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(
    query_features,
    gallery_features,
    distmat,
    query=None,
    gallery=None,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    cmc_topk=(1, 5, 10),
    cmc_flag=False,
    regdb=False,
):
    del query_features, gallery_features
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert query_ids is not None and gallery_ids is not None
        assert query_cams is not None and gallery_cams is not None

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, regdb=regdb)
    print("Mean AP: {:4.1%}".format(mAP))

    if not cmc_flag:
        return mAP

    cmc_configs = {
        "market1501": dict(separate_camera_set=False, single_gallery_shot=False, first_match_break=True),
    }
    cmc_scores = {
        name: cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, regdb=regdb, **params)
        for name, params in cmc_configs.items()
    }

    print("CMC Scores:")
    for k in cmc_topk:
        print("  top-{:<4}{:12.1%}".format(k, cmc_scores["market1501"][k - 1]))
    return cmc_scores["market1501"], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False, modal=0, regdb=False):
        features, _, _ = extract_features(self.model, data_loader, mode=modal)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, regdb=regdb)

        if not rerank:
            return results

        print("Applying person re-ranking ...")
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)


class ImageListDataset(data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_paths)


def build_test_transform(args_or_height, width=None):
    if width is None:
        args = args_or_height
        height = args.height
        width = args.width
        args.img_h = height
        args.img_w = width
    else:
        height = args_or_height

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer,
    ])


def _build_loader_kwargs(workers, pin_memory=True, persistent_workers=False):
    kwargs = {"num_workers": workers, "pin_memory": pin_memory}
    if workers > 0 and persistent_workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def build_eval_loader(image_paths, labels, batch_size, workers, height, width, transform=None):
    transform = transform or build_test_transform(height, width)
    dataset = ImageListDataset(image_paths, labels, transform=transform)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_build_loader_kwargs(workers),
    )


def extract_eval_features(model, loader, modal, desc=None, runtime_args=None):
    model.eval()
    features = []
    if desc is not None:
        print("Extracting {} Feature...".format(desc))

    start = time.time()
    with torch.inference_mode():
        for inputs, _ in loader:
            inputs = inputs.cuda(non_blocking=True)
            flipped = fliplr(inputs)
            with get_inference_autocast_context(runtime_args):
                outputs = model(inputs, inputs, modal)
                outputs_flip = model(flipped, flipped, modal)
            merged = (outputs.detach() + outputs_flip.detach()) / 2.0
            merged = merged.float()
            merged = merged / torch.norm(merged, p=2, dim=1, keepdim=True)
            features.append(merged.cpu())

    print("Extracting Time:\t {:.3f}".format(time.time() - start))
    if not features:
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(features, dim=0).numpy()


def extract_gall_feat(model, gall_loader, ngall, runtime_args=None):
    del ngall
    return extract_eval_features(model, gall_loader, modal=1, desc="Gallery", runtime_args=runtime_args)


def extract_query_feat(model, query_loader, nquery, runtime_args=None):
    del nquery
    return extract_eval_features(model, query_loader, modal=2, desc="Query", runtime_args=runtime_args)


def pairwise_similarity(query_features, gallery_features):
    return np.matmul(query_features, np.transpose(gallery_features))


@lru_cache(maxsize=None)
def process_query_sysu(data_path, mode="all"):
    if mode not in ("all", "indoor"):
        raise ValueError("Unsupported SYSU mode: {}".format(mode))

    file_path = os.path.join(data_path, "exp", "test_id.txt")
    with open(file_path, "r") as handle:
        ids = ["%04d" % int(value) for value in handle.read().splitlines()[0].split(",")]

    query_images = []
    query_ids = []
    query_cams = []
    for person_id in sorted(ids):
        for camera in ("cam3", "cam6"):
            image_dir = os.path.join(data_path, camera, person_id)
            if not os.path.isdir(image_dir):
                continue
            for name in sorted(os.listdir(image_dir)):
                path = os.path.join(image_dir, name)
                query_images.append(path)
                query_ids.append(int(path[-13:-9]))
                query_cams.append(int(path[-15]))
    return query_images, np.array(query_ids), np.array(query_cams)


@lru_cache(maxsize=None)
def process_gallery_sysu(data_path, mode="all", trial=0):
    rng = random.Random(trial)
    if mode == "all":
        cameras = ("cam1", "cam2", "cam4", "cam5")
    elif mode == "indoor":
        cameras = ("cam1", "cam2")
    else:
        raise ValueError("Unsupported SYSU mode: {}".format(mode))

    file_path = os.path.join(data_path, "exp", "test_id.txt")
    with open(file_path, "r") as handle:
        ids = ["%04d" % int(value) for value in handle.read().splitlines()[0].split(",")]

    gallery_images = []
    gallery_ids = []
    gallery_cams = []
    for person_id in sorted(ids):
        for camera in cameras:
            image_dir = os.path.join(data_path, camera, person_id)
            if not os.path.isdir(image_dir):
                continue
            candidates = sorted(os.listdir(image_dir))
            if not candidates:
                continue
            path = os.path.join(image_dir, rng.choice(candidates))
            gallery_images.append(path)
            gallery_ids.append(int(path[-13:-9]))
            gallery_cams.append(int(path[-15]))
    return gallery_images, np.array(gallery_ids), np.array(gallery_cams)


def process_test_regdb(data_path, trial=1, modal="visible"):
    filename = "test_visible_{}.txt".format(trial)
    if modal == "thermal":
        filename = "test_thermal_{}.txt".format(trial)
    file_path = os.path.join(data_path, "idx", filename)
    with open(file_path, "rt") as handle:
        rows = handle.read().splitlines()
    image_paths = [os.path.join(data_path, row.split(" ")[0]) for row in rows]
    labels = [int(row.split(" ")[1]) for row in rows]
    return image_paths, np.array(labels)


def process_query_llcm(data_path, mode=1):
    if mode == 1:
        cameras = ("test_vis/cam1", "test_vis/cam2", "test_vis/cam3", "test_vis/cam4", "test_vis/cam5", "test_vis/cam6", "test_vis/cam7", "test_vis/cam8", "test_vis/cam9")
    elif mode == 2:
        cameras = ("test_nir/cam1", "test_nir/cam2", "test_nir/cam4", "test_nir/cam5", "test_nir/cam6", "test_nir/cam7", "test_nir/cam8", "test_nir/cam9")
    else:
        raise ValueError("Unsupported LLCM mode: {}".format(mode))

    file_path = os.path.join(data_path, "idx", "test_id.txt")
    with open(file_path, "r") as handle:
        ids = ["%04d" % int(value) for value in handle.read().splitlines()[0].split(",")]

    query_images = []
    query_ids = []
    query_cams = []
    for person_id in sorted(ids):
        for camera in cameras:
            image_dir = os.path.join(data_path, camera, person_id)
            if not os.path.isdir(image_dir):
                continue
            for name in sorted(os.listdir(image_dir)):
                path = os.path.join(image_dir, name)
                query_images.append(path)
                query_ids.append(int(path.split("cam")[1][2:6]))
                query_cams.append(int(path.split("cam")[1][0]))
    return query_images, np.array(query_ids), np.array(query_cams)


def process_gallery_llcm(data_path, mode=1, trial=0):
    rng = random.Random(trial)
    if mode == 2:
        cameras = ("test_vis/cam1", "test_vis/cam2", "test_vis/cam3", "test_vis/cam4", "test_vis/cam5", "test_vis/cam6", "test_vis/cam7", "test_vis/cam8", "test_vis/cam9")
    elif mode == 1:
        cameras = ("test_nir/cam1", "test_nir/cam2", "test_nir/cam4", "test_nir/cam5", "test_nir/cam6", "test_nir/cam7", "test_nir/cam8", "test_nir/cam9")
    else:
        raise ValueError("Unsupported LLCM mode: {}".format(mode))

    file_path = os.path.join(data_path, "idx", "test_id.txt")
    with open(file_path, "r") as handle:
        ids = ["%04d" % int(value) for value in handle.read().splitlines()[0].split(",")]

    gallery_images = []
    gallery_ids = []
    gallery_cams = []
    for person_id in sorted(ids):
        for camera in cameras:
            image_dir = os.path.join(data_path, camera, person_id)
            if not os.path.isdir(image_dir):
                continue
            candidates = sorted(os.listdir(image_dir))
            if not candidates:
                continue
            path = os.path.join(image_dir, rng.choice(candidates))
            gallery_images.append(path)
            gallery_ids.append(int(path.split("cam")[1][2:6]))
            gallery_cams.append(int(path.split("cam")[1][0]))
    return gallery_images, np.array(gallery_ids), np.array(gallery_cams)


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    unique_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        dedup = pred_label[q_idx][keep]
        first_index = np.unique(dedup, return_index=True)[1]
        dedup = [dedup[index] for index in sorted(first_index)]
        unique_cmc.append((np.asarray(dedup) == q_pid).astype(np.int32).cumsum()[:max_rank])

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc_values = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        all_inp.append(cmc_values[pos_max_idx] / (pos_max_idx + 1.0))
        num_valid_q += 1.0

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = np.asarray([value / (index + 1.0) for index, value in enumerate(tmp_cmc)]) * raw_cmc
        all_ap.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    unique_cmc = np.asarray(unique_cmc).astype(np.float32).sum(0) / num_valid_q
    return unique_cmc, np.mean(all_ap), np.mean(all_inp)


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    all_inp = []
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc_values = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        all_inp.append(cmc_values[pos_max_idx] / (pos_max_idx + 1.0))
        cmc_values[cmc_values > 1] = 1
        all_cmc.append(cmc_values[:max_rank])
        num_valid_q += 1.0

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = np.asarray([value / (index + 1.0) for index, value in enumerate(tmp_cmc)]) * raw_cmc
        all_ap.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    return all_cmc, np.mean(all_ap), np.mean(all_inp)


def eval_llcm(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    unique_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        dedup = pred_label[q_idx][keep]
        first_index = np.unique(dedup, return_index=True)[1]
        dedup = [dedup[index] for index in sorted(first_index)]
        unique_cmc.append((np.asarray(dedup) == q_pid).astype(np.int32).cumsum()[:max_rank])

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc_values = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        all_inp.append(cmc_values[pos_max_idx] / (pos_max_idx + 1.0))
        num_valid_q += 1.0

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = np.asarray([value / (index + 1.0) for index, value in enumerate(tmp_cmc)]) * raw_cmc
        all_ap.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    unique_cmc = np.asarray(unique_cmc).astype(np.float32).sum(0) / num_valid_q
    return unique_cmc, np.mean(all_ap), np.mean(all_inp)


def _print_metrics(prefix, cmc_values, mAP, mINP):
    print(
        "{}Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
            prefix,
            cmc_values[0],
            cmc_values[4],
            cmc_values[9],
            cmc_values[19],
            mAP,
            mINP,
        )
    )


def print_result(title, cmc_values, mAP, mINP):
    print(title)
    _print_metrics("FC:   ", cmc_values, mAP, mINP)


def _average_result(cmc_total, mAP_total, mINP_total, divisor):
    return {
        "cmc": cmc_total / float(divisor),
        "mAP": mAP_total / float(divisor),
        "mINP": mINP_total / float(divisor),
    }


def evaluate_sysu_mode(model, args, transform_test=None, mode="all"):
    transform_test = transform_test or build_test_transform(args)
    query_img, query_label, query_cam = process_query_sysu(args.data_dir, mode=mode)
    query_loader = build_eval_loader(
        query_img,
        query_label,
        args.test_batch,
        args.workers,
        args.img_h,
        args.img_w,
        transform=transform_test,
    )
    query_feat = extract_query_feat(model, query_loader, len(query_label), runtime_args=args)

    cmc_total = None
    mAP_total = 0.0
    mINP_total = 0.0
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_dir, mode=mode, trial=trial)
        gall_loader = build_eval_loader(
            gall_img,
            gall_label,
            args.test_batch,
            args.workers,
            args.img_h,
            args.img_w,
            transform=transform_test,
        )
        gall_feat = extract_gall_feat(model, gall_loader, len(gall_label), runtime_args=args)
        cmc_values, mAP, mINP = eval_sysu(-pairwise_similarity(query_feat, gall_feat), query_label, gall_label, query_cam, gall_cam)
        cmc_total = cmc_values if cmc_total is None else cmc_total + cmc_values
        mAP_total += mAP
        mINP_total += mINP

        print("Test Trial: {}".format(trial))
        _print_metrics("FC:   ", cmc_values, mAP, mINP)

    result = _average_result(cmc_total, mAP_total, mINP_total, 10)
    print("All Average:")
    _print_metrics("FC:     ", result["cmc"], result["mAP"], result["mINP"])
    return result["mAP"]


def evaluate_regdb_direction(model, args, transform_test=None, trial=1, query_modal="visible", gallery_modal="thermal", title=None):
    transform_test = transform_test or build_test_transform(args)
    query_img, query_label = process_test_regdb(args.data_dir, trial=trial, modal=query_modal)
    gall_img, gall_label = process_test_regdb(args.data_dir, trial=trial, modal=gallery_modal)

    query_loader = build_eval_loader(
        query_img,
        query_label,
        args.test_batch,
        args.workers,
        args.img_h,
        args.img_w,
        transform=transform_test,
    )
    gall_loader = build_eval_loader(
        gall_img,
        gall_label,
        args.test_batch,
        args.workers,
        args.img_h,
        args.img_w,
        transform=transform_test,
    )
    query_feat = extract_query_feat(model, query_loader, len(query_label), runtime_args=args)
    gall_feat = extract_gall_feat(model, gall_loader, len(gall_label), runtime_args=args)
    cmc_values, mAP, mINP = eval_regdb(-pairwise_similarity(query_feat, gall_feat), query_label, gall_label)

    if title is not None:
        print(title)
    _print_metrics("FC:   ", cmc_values, mAP, mINP)
    return mAP


def evaluate_llcm_mode(model, args, transform_test=None, mode=1):
    transform_test = transform_test or build_test_transform(args)
    query_img, query_label, query_cam = process_query_llcm(args.data_dir, mode=mode)
    query_loader = build_eval_loader(
        query_img,
        query_label,
        args.test_batch,
        args.workers,
        args.img_h,
        args.img_w,
        transform=transform_test,
    )
    query_feat = extract_query_feat(model, query_loader, len(query_label), runtime_args=args)

    cmc_total = None
    mAP_total = 0.0
    mINP_total = 0.0
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(args.data_dir, mode=mode, trial=trial)
        gall_loader = build_eval_loader(
            gall_img,
            gall_label,
            args.test_batch,
            args.workers,
            args.img_h,
            args.img_w,
            transform=transform_test,
        )
        gall_feat = extract_gall_feat(model, gall_loader, len(gall_label), runtime_args=args)
        cmc_values, mAP, mINP = eval_llcm(-pairwise_similarity(query_feat, gall_feat), query_label, gall_label, query_cam, gall_cam)
        cmc_total = cmc_values if cmc_total is None else cmc_total + cmc_values
        mAP_total += mAP
        mINP_total += mINP

        print("Test Trial: {}".format(trial))
        _print_metrics("FC:   ", cmc_values, mAP, mINP)

    result = _average_result(cmc_total, mAP_total, mINP_total, 10)
    print("All Average:")
    _print_metrics("FC:     ", result["cmc"], result["mAP"], result["mINP"])
    return result["mAP"]


def evaluate_sysu(model, args):
    results = {}
    for mode in ("all", "indoor"):
        build_test_transform(args)
        query_img, query_label, query_cam = process_query_sysu(args.data_dir, mode=mode)
        query_loader = build_eval_loader(query_img, query_label, args.test_batch, args.workers, args.img_h, args.img_w)
        query_feat = extract_query_feat(model, query_loader, len(query_label), runtime_args=args)

        cmc_total = None
        mAP_total = 0.0
        mINP_total = 0.0
        for trial in range(10):
            gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_dir, mode=mode, trial=trial)
            gall_loader = build_eval_loader(gall_img, gall_label, args.test_batch, args.workers, args.img_h, args.img_w)
            gall_feat = extract_gall_feat(model, gall_loader, len(gall_label), runtime_args=args)
            cmc_values, mAP, mINP = eval_sysu(-pairwise_similarity(query_feat, gall_feat), query_label, gall_label, query_cam, gall_cam)
            cmc_total = cmc_values if cmc_total is None else cmc_total + cmc_values
            mAP_total += mAP
            mINP_total += mINP

        results[mode] = _average_result(cmc_total, mAP_total, mINP_total, 10)
        print_result("SYSU {}".format(mode), results[mode]["cmc"], results[mode]["mAP"], results[mode]["mINP"])
    return results


def evaluate_regdb(model, args):
    results = {}
    build_test_transform(args)
    trials = list(range(1, 11)) if getattr(args, "all_trials", False) else [args.trial]
    directions = (("visible", "thermal"), ("thermal", "visible"))
    for query_modal, gallery_modal in directions:
        cmc_total = None
        mAP_total = 0.0
        mINP_total = 0.0
        for trial in trials:
            query_img, query_label = process_test_regdb(args.data_dir, trial=trial, modal=query_modal)
            gall_img, gall_label = process_test_regdb(args.data_dir, trial=trial, modal=gallery_modal)
            query_loader = build_eval_loader(query_img, query_label, args.test_batch, args.workers, args.img_h, args.img_w)
            gall_loader = build_eval_loader(gall_img, gall_label, args.test_batch, args.workers, args.img_h, args.img_w)
            query_feat = extract_query_feat(model, query_loader, len(query_label), runtime_args=args)
            gall_feat = extract_gall_feat(model, gall_loader, len(gall_label), runtime_args=args)
            cmc_values, mAP, mINP = eval_regdb(-pairwise_similarity(query_feat, gall_feat), query_label, gall_label)
            cmc_total = cmc_values if cmc_total is None else cmc_total + cmc_values
            mAP_total += mAP
            mINP_total += mINP

        key = "{}_to_{}".format(query_modal, gallery_modal)
        results[key] = _average_result(cmc_total, mAP_total, mINP_total, len(trials))
        print_result("RegDB {}".format(key), results[key]["cmc"], results[key]["mAP"], results[key]["mINP"])
    return results


def evaluate_llcm(model, args):
    results = {}
    for mode, title in ((1, "vis_to_nir"), (2, "nir_to_vis")):
        build_test_transform(args)
        query_img, query_label, query_cam = process_query_llcm(args.data_dir, mode=mode)
        query_loader = build_eval_loader(query_img, query_label, args.test_batch, args.workers, args.img_h, args.img_w)
        query_feat = extract_query_feat(model, query_loader, len(query_label), runtime_args=args)

        cmc_total = None
        mAP_total = 0.0
        mINP_total = 0.0
        for trial in range(10):
            gall_img, gall_label, gall_cam = process_gallery_llcm(args.data_dir, mode=mode, trial=trial)
            gall_loader = build_eval_loader(gall_img, gall_label, args.test_batch, args.workers, args.img_h, args.img_w)
            gall_feat = extract_gall_feat(model, gall_loader, len(gall_label), runtime_args=args)
            cmc_values, mAP, mINP = eval_llcm(-pairwise_similarity(query_feat, gall_feat), query_label, gall_label, query_cam, gall_cam)
            cmc_total = cmc_values if cmc_total is None else cmc_total + cmc_values
            mAP_total += mAP
            mINP_total += mINP

        results[title] = _average_result(cmc_total, mAP_total, mINP_total, 10)
        print_result("LLCM {}".format(title), results[title]["cmc"], results[title]["mAP"], results[title]["mINP"])
    return results


def create_model(args):
    model = models.create(
        args.arch,
        num_features=args.features,
        norm=True,
        dropout=args.dropout,
        num_classes=0,
        pooling_type=args.pooling_type,
    )
    model.cuda()
    return nn.DataParallel(model)


def run_evaluation(args):
    model = create_model(args)
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if args.dataset == "sysu":
        return evaluate_sysu(model, args)
    if args.dataset == "regdb":
        return evaluate_regdb(model, args)
    return evaluate_llcm(model, args)
