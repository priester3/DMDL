# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.dmdl.acceleration import build_optimizer, log_acceleration_config, wrap_model_for_training
from clustercontrast.dmdl.parser import build_legacy_dataset_parser, detect_stage_from_cli, require_dataset_data_dir
from clustercontrast.models.cm_topk import ClusterMemory as ClusterMemory_Single
from clustercontrast.trainers import CausalTrainer_s1_woCAM
from clustercontrast.evaluators import build_test_transform, evaluate_sysu_mode, extract_features_tensor
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color, Preprocessor_ir
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam

import math
import timm
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange
from ChannelAug import PseudoColor_new as PseudoColor

from clustercontrast.models.cm_causal_softtopk import ClusterMemory as Stage2ClusterMemory
from clustercontrast.trainers import CausalTrainer_s2_woCAM, CausalTrainer_s2_woCAM_inter
from clustercontrast.utils.data import IterLoader, IterLoader_Inter
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color, Preprocessor_ir, Preprocessor_inter
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam, RandomMultipleGallerySamplerInterModality

_LOADER_RUNTIME_ARGS = None
_WORKING_DIR = osp.dirname(osp.abspath(__file__))
_PARSER_COMMON_DEFAULTS = {
    "data_dir_default": None,
    "logs_dir_default": osp.join(_WORKING_DIR, "logs_sysu"),
    "batch_size_default": 256,
    "num_instances_default": 16,
    "epochs_default": 50,
    "iters_default": 200,
    "workers_default": 8,
    "arch_default": "agw_one",
    "k1_default": 30,
    "seed_default": 1,
    "log_s1_default": "sysu_s1/dmdl",
    "log_s2_default": "sysu_s2/dmdl",
}
_PARSER_STAGE_DEFAULTS = {
    1: {
        "include_topk": True,
        "topk_default": 3,
    },
    2: {
        "include_lmada": True,
        "lmada_default": 1.0,
        "include_topk": True,
        "topk_default": 10,
    },
}


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def build_loader_kwargs(workers, pin_memory=True, drop_last=False):
    kwargs = {
        'num_workers': workers,
        'pin_memory': getattr(_LOADER_RUNTIME_ARGS, 'pin_memory', pin_memory),
        'drop_last': drop_last,
    }
    if workers > 0:
        kwargs['persistent_workers'] = getattr(_LOADER_RUNTIME_ARGS, 'persistent_workers', True)
        prefetch_factor = getattr(_LOADER_RUNTIME_ARGS, 'prefetch_factor', 2)
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor
    return kwargs


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):
    
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor_ir(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, sampler=sampler,
                   shuffle=not rmgs_flag, **build_loader_kwargs(workers, drop_last=True)), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, sampler=sampler,
                       shuffle=not rmgs_flag, **build_loader_kwargs(workers, drop_last=True)), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, sampler=sampler,
                       shuffle=not rmgs_flag, **build_loader_kwargs(workers, drop_last=True)), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size,
        shuffle=False, **build_loader_kwargs(workers))

    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    return wrap_model_for_training(model, args)



def _stage2_get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def _stage2_build_loader_kwargs(workers, pin_memory=True, drop_last=False):
    kwargs = {
        'num_workers': workers,
        'pin_memory': getattr(_LOADER_RUNTIME_ARGS, 'pin_memory', pin_memory),
        'drop_last': drop_last,
    }
    if workers > 0:
        kwargs['persistent_workers'] = getattr(_LOADER_RUNTIME_ARGS, 'persistent_workers', True)
        prefetch_factor = getattr(_LOADER_RUNTIME_ARGS, 'prefetch_factor', 2)
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor
    return kwargs


def _stage2_get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                   batch_size=batch_size, sampler=sampler,
                   shuffle=not rmgs_flag, **_stage2_build_loader_kwargs(workers, drop_last=True)), length=iters)

    return train_loader


def _stage2_get_train_loader_inter(args, dataset_rgb, dataset_ir, height, width, batch_size, workers,
                     num_instances, iters, trainset_rgb=None, trainset_ir=None, no_cam=False,train_transformer_rgb=None,train_transformer_rgb1=None,
                     train_transformer_ir=None,train_transformer_ir1=None):


    train_set_rgb = sorted(dataset_rgb.train) if trainset_rgb is None else sorted(trainset_rgb)
    train_set_ir = sorted(dataset_ir.train) if trainset_ir is None else sorted(trainset_ir)
    
    rmgs_flag = num_instances > 0
    sampler = RandomMultipleGallerySamplerInterModality(train_set_rgb, train_set_ir, num_instances)
    
    train_set = Preprocessor_inter(train_set_rgb, train_set_ir, root_rgb=dataset_rgb.images_dir, root_ir=dataset_ir.images_dir, 
                               transform_rgb=train_transformer_rgb, transform1_rgb=train_transformer_rgb1, 
                               transform_ir=train_transformer_ir, transform1_ir=train_transformer_ir1)
    train_set.inedex_rgb = sampler.index_rgb
    train_set.inedex_ir = sampler.index_ir

    train_loader = IterLoader_Inter(
            DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                       shuffle=not rmgs_flag, **_stage2_build_loader_kwargs(workers, drop_last=True)), length=iters)

    return train_loader


def _stage2_get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size,
        shuffle=False, **_stage2_build_loader_kwargs(workers))

    return test_loader

def _stage2_create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    return wrap_model_for_training(model, args)


def _stage2_load_checkpoint_into_model(model, checkpoint, tag):
    model_state = model.state_dict()
    checkpoint_state = checkpoint['state_dict']
    compatible_state = {}
    skipped_keys = []

    for key, value in checkpoint_state.items():
        if key not in model_state or model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    if skipped_keys:
        print("=> Ignoring incompatible checkpoint keys for {}: {}".format(
            tag, ", ".join(skipped_keys[:10])
        ))

    missing_keys = sorted(set(model_state.keys()) - set(compatible_state.keys()))
    if missing_keys:
        print("=> Missing model keys after checkpoint filtering for {}: {}".format(
            tag, ", ".join(missing_keys[:10])
        ))

    model.load_state_dict(compatible_state, strict=False)

def run_stage1(args=None):
    if args is None:
        args = _build_stage1_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    global _LOADER_RUNTIME_ARGS
    _LOADER_RUNTIME_ARGS = args
    start_epoch=0
    best_mAP=0
    stage1_logs_dir = osp.join(args.logs_dir+'/'+args.log_s1_name)
    start_time = time.monotonic()

    # cudnn.benchmark = True
    
    sys.stdout = Logger(osp.join(stage1_logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    log_acceleration_config(args)
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)
    sorted_ir_train = sorted(dataset_ir.train)
    sorted_rgb_train = sorted(dataset_rgb.train)

    # Create model
    model = create_model(args)
    # Optimizer
    optimizer = build_optimizer(model, args)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # Trainer
    trainer = CausalTrainer_s1_woCAM(model)
    # ########################
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    height=args.height
    width=args.width
    train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        # ChannelRandomErasing(probability = 0.5)
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                                max_area=0.5,mode='pixel',
                                                max_count=1,device='cpu'),
    ])
    
    train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
        T.ToTensor(),
        normalizer,
        # ChannelRandomErasing(probability = 0.5),
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                                max_area=0.5,mode='pixel',
                                                max_count=1,device='cpu'),
        ChannelExchange(gray = 2)#2, TODO
    ])

    transform_thermal = T.Compose( [
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        # ChannelRandomErasing(probability = 0.5),
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                                max_area=0.5,mode='pixel',
                                                max_count=1,device='cpu'),
        ChannelAdapGray(probability =0.5)
    ])
    
    transform_thermal1 = T.Compose( [
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        PseudoColor(),
        T.ToTensor(),
        normalizer,
        # ChannelRandomErasing(probability = 0.5),
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                                max_area=0.5,mode='pixel',
                                                max_count=1,device='cpu'),
    ])
    transform_test = build_test_transform(args)
    cluster_loader_rgb = get_test_loader(
        dataset_rgb,
        args.height,
        args.width,
        args.batch_size,
        args.workers,
        testset=sorted_rgb_train,
    )
    cluster_loader_ir = get_test_loader(
        dataset_ir,
        args.height,
        args.width,
        args.batch_size,
        args.workers,
        testset=sorted_ir_train,
    )
    
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = args.eps
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.eps
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            features_rgb, cids_rgb = extract_features_tensor(
                model, cluster_loader_rgb, print_freq=50, mode=1, runtime_args=args, keep_on_gpu=True
            )
            cids_rgb = cids_rgb.numpy()

            print('==> Create pseudo labels for unlabeled IR data')
            features_ir, cids_ir = extract_features_tensor(
                model, cluster_loader_ir, print_freq=50, mode=2, runtime_args=args, keep_on_gpu=True
            )
            cids_ir = cids_ir.numpy()
            

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,
                                                      search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,
                                                       search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)  # cluster rgb by using aug feature 
            del rerank_dist_rgb
            del rerank_dist_ir

            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)  # using raw feature represent cluster
        memory_ir = ClusterMemory_Single(model.module.num_features, num_cluster_ir, topk= args.topk, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory_Single(model.module.num_features, num_cluster_rgb, topk= args.topk, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        del cluster_features_rgb, cluster_features_ir

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb


        # Computer GMM for IR/RGB clusters to achieve denoising
        if epoch < 30:
            confidence_ir = np.ones((pseudo_labels_ir.shape[0],))
            confidence_rgb = np.ones((pseudo_labels_rgb.shape[0],))
        else:
            features_ir = F.normalize(features_ir, dim=1)
            # Keep matmul operands on the same device (features can be CUDA after clustering optimizations).
            context_assignments_logits_ir = features_ir.mm(memory_ir.features.T.to(features_ir.device)) / 0.05
            context_assignments_ir = F.softmax(context_assignments_logits_ir, dim=1)

            losses_ir = - context_assignments_ir[torch.arange(pseudo_labels_ir.shape[0]), pseudo_labels_ir]
            losses_ir = losses_ir.cpu().numpy()[:, np.newaxis]
            losses_ir = (losses_ir - losses_ir.min()) / (losses_ir.max() - losses_ir.min())

            confidence_ir = np.zeros((losses_ir.shape[0],))
            for i in range(num_cluster_ir):
                mask = pseudo_labels_ir == i
                if mask.sum() <= 1.:
                    confidence_ir[mask] = 1.
                    print(i)
                    continue
                c = losses_ir[mask, :]
                gm = GaussianMixture(n_components=2, random_state=0).fit(c)
                pdf = gm.predict_proba(c)
                confidence_ir[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]

            features_rgb = F.normalize(features_rgb, dim=1)
            context_assignments_logits_rgb = features_rgb.mm(memory_rgb.features.T.to(features_rgb.device)) / 0.05
            context_assignments_rgb = F.softmax(context_assignments_logits_rgb, dim=1)

            losses_rgb = - context_assignments_rgb[torch.arange(pseudo_labels_rgb.shape[0]), pseudo_labels_rgb]
            losses_rgb = losses_rgb.cpu().numpy()[:, np.newaxis]
            losses_rgb = (losses_rgb - losses_rgb.min()) / (losses_rgb.max() - losses_rgb.min())

            confidence_rgb = np.zeros((losses_rgb.shape[0],))
            for i in range(num_cluster_rgb):
                mask = pseudo_labels_rgb == i
                if mask.sum() <= 1.:
                    confidence_rgb[mask] = 1.
                    print(i)
                    continue
                c = losses_rgb[mask, :]
                gm = GaussianMixture(n_components=2, random_state=0).fit(c)
                pdf = gm.predict_proba(c)
                confidence_rgb[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]


        pseudo_labeled_dataset_ir = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted_ir_train, pseudo_labels_ir)):
            conf = confidence_ir[i]
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), conf.item(), cid))
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))
        pseudo_labeled_dataset_rgb = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted_rgb_train, pseudo_labels_rgb)):
            conf = confidence_rgb[i]
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), conf.item(), cid))
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))


        train_loader_ir = get_train_loader_color(args, dataset_ir, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal,train_transformer1=transform_thermal1)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
        

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        
        trainer.train(args, epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            mAP = evaluate_sysu_mode(model, args, transform_test, mode='all')
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(stage1_logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()

    print('==> Test with the best model all search:')
    checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    transform_test = build_test_transform(args)
    evaluate_sysu_mode(model, args, transform_test, mode='all')
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

    print('==> Test with the best model indoor search:')
    checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluate_sysu_mode(model, args, transform_test, mode='indoor')
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


def run_stage2(args=None):
    if args is None:
        args = _build_stage2_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    global _LOADER_RUNTIME_ARGS
    _LOADER_RUNTIME_ARGS = args
    start_epoch=0
    best_mAP=0
    stage2_logs_dir = osp.join(args.logs_dir+'/'+args.log_s2_name)
    start_time = time.monotonic()

    # cudnn.benchmark = True

    sys.stdout = Logger(osp.join(stage2_logs_dir, 'log.txt'))
    print(stage2_logs_dir)
    print("using camen; semi-cross-modality update; single loss in single-modality loss; change miu 0.25/0.5")
    print("==========\nArgs:{}\n==========".format(args))
    log_acceleration_config(args)

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = _stage2_get_data('sysu_ir', args.data_dir)
    dataset_rgb = _stage2_get_data('sysu_rgb', args.data_dir)
    sorted_ir_train = sorted(dataset_ir.train)
    sorted_rgb_train = sorted(dataset_rgb.train)

    # Create model
    checkpoint = load_checkpoint(osp.join(args.logs_dir + '/' + args.log_s1_name, 'model_best.pth.tar'))
    # checkpoint = load_checkpoint('/mnt/workspace/lijiaze/usl-new/logs_one/s1/1-agwone-cam/model_best.pth.tar')
    model = _stage2_create_model(args)
    _stage2_load_checkpoint_into_model(model, checkpoint, 'stage1 init')
    
    # trainer = CausalTrainer_s2_woCAM(model)
    
    # Optimizer
    optimizer = build_optimizer(model, args)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    transform_test = build_test_transform(args)
    cluster_loader_rgb = _stage2_get_test_loader(
        dataset_rgb,
        args.height,
        args.width,
        args.batch_size,
        args.workers,
        testset=sorted_rgb_train,
    )
    cluster_loader_ir = _stage2_get_test_loader(
        dataset_ir,
        args.height,
        args.width,
        args.batch_size,
        args.workers,
        testset=sorted_ir_train,
    )
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    height=args.height
    width=args.width
    train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                            max_area=0.5,mode='pixel',
                                            max_count=1,device='cpu'),
    ])

    train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
        T.ToTensor(),
        normalizer,
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                            max_area=0.5,mode='pixel',
                                            max_count=1,device='cpu'),
        ChannelExchange(gray = 2),
    ])

    transform_thermal = T.Compose( [
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                            max_area=0.5,mode='pixel',
                                            max_count=1,device='cpu'),
        ChannelAdapGray(probability =0.5)
    ])

    transform_thermal1 = T.Compose( [
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        PseudoColor(),
        T.ToTensor(),
        normalizer,
        timm.data.random_erasing.RandomErasing(probability=0.5, min_area=0.02,
                                            max_area=0.5,mode='pixel',
                                            max_count=1,device='cpu'),
    ])

    for epoch in range(args.epochs):
        if epoch < 40:
            trainer = CausalTrainer_s2_woCAM(model)
        else:
            trainer = CausalTrainer_s2_woCAM_inter(model)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers_id = [idx for idx in sorted(centers.keys())]
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers, centers_id
        
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = args.eps
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.eps
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            features_rgb, cids_rgb = extract_features_tensor(
                model, cluster_loader_rgb, print_freq=50, mode=1, runtime_args=args, keep_on_gpu=True
            )
            cids_rgb = cids_rgb.numpy()

            print('==> Create pseudo labels for unlabeled IR data')
            features_ir, cids_ir = extract_features_tensor(
                model, cluster_loader_ir, print_freq=50, mode=2, runtime_args=args, keep_on_gpu=True
            )
            cids_ir = cids_ir.numpy()
            

            ## add rgb_k1
            if epoch % 2 == 0:
                ir_k1 = args.k1
                rgb_k1 = 15
            else:
                ir_k1 = 15
                rgb_k1 = args.k1
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=ir_k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=rgb_k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)  # cluster rgb by using aug feature 
            del rerank_dist_rgb
            del rerank_dist_ir

            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        # generate new dataset and calculate cluster centers
        

        cluster_features_ir, _ = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb, _ = generate_cluster_features(pseudo_labels_rgb, features_rgb)

        print("Max Matching")
        cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)   # V × L
        cluster_features_ir = F.normalize(cluster_features_ir, dim=1)  # I × L
        # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
        similarity = ((torch.mm(cluster_features_rgb, cluster_features_ir.T))/1).exp().cpu() #.exp().cpu() V × I
        dis_similarity = (1 / (similarity))
        cost = dis_similarity / 1  # V × I
        
        r2i_list = torch.sort(cost)[1][:,0].tolist()
        r2i = dict(enumerate(r2i_list))

        i2r_list = torch.sort(cost.T)[1][:,0].tolist()
        i2r = dict(enumerate(i2r_list))

        print("Max Matching Done")

        r2i[-1]=-1
        i2r[-1]=-1
        if epoch % 2 == 0:
            pseudo_labels_rgb_new = np.array([r2i[key.item()] for key in pseudo_labels_rgb])
            pseudo_labels_ir_new = pseudo_labels_ir
            num_cluster_all = num_cluster_ir
        else:
            pseudo_labels_rgb_new = pseudo_labels_rgb
            pseudo_labels_ir_new = np.array([i2r[key.item()] for key in pseudo_labels_ir])
            num_cluster_all = num_cluster_rgb

        del cluster_features_ir, cluster_features_rgb

        cluster_features_rgb, pids_rgb = generate_cluster_features(pseudo_labels_rgb_new, features_rgb)
        cluster_features_ir, pids_ir = generate_cluster_features(pseudo_labels_ir_new, features_ir)
        memory_all = Stage2ClusterMemory(model.module.num_features, len(pids_rgb), len(pids_ir), num_cluster_all, topk=args.topk, temp=args.temp,
                            momentum=args.momentum).cuda()
        memory_all.features_rgb = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory_all.features_ir = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_all.pids_rgb = torch.Tensor(pids_rgb).long().cuda()
        memory_all.pids_ir = torch.Tensor(pids_ir).long().cuda()

        labels_all = np.concatenate((pseudo_labels_rgb_new,pseudo_labels_ir_new),axis=0)
        features_all = torch.cat((features_rgb,features_ir),dim=0)
        cluster_features_all, _ = generate_cluster_features(labels_all, features_all)

        memory_all.features_all = F.normalize(cluster_features_all, dim=1).cuda()
        trainer.memory_all = memory_all

        features_all = F.normalize(features_all, dim=1)
        context_assignments_logits = features_all.mm(memory_all.features_all.T.to(features_all.device)) / 0.05
        context_assignments = F.softmax(context_assignments_logits, dim=1)

        losses = - context_assignments[torch.arange(labels_all.shape[0]), labels_all]
        losses = losses.cpu().numpy()[:, np.newaxis]
        losses = (losses - losses.min()) / (losses.max() - losses.min())

        confidence_all = np.zeros((losses.shape[0],))
        for i in range(num_cluster_all):
            mask = labels_all == i
            if mask.sum() <= 1.:
                confidence_all[mask] = 1.
                print(i)
                continue
            c = losses[mask, :]
            gm = GaussianMixture(n_components=2, random_state=0).fit(c)
            pdf = gm.predict_proba(c)
            confidence_all[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]

        confidence_rgb = confidence_all[:pseudo_labels_rgb_new.shape[0]]
        confidence_ir = confidence_all[pseudo_labels_rgb_new.shape[0]:]

        print("mean conf (ir):", confidence_ir.mean())
        print("mean conf (rgb):", confidence_rgb.mean())

        pseudo_labeled_dataset_ir = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted_ir_train, pseudo_labels_ir_new)):
            conf = confidence_ir[i]
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), conf.item(), cid))
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))
        pseudo_labeled_dataset_rgb = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted_rgb_train, pseudo_labels_rgb_new)):
            conf = confidence_rgb[i]
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), conf.item(), cid))
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ###############################
        rgb_instance_num = len(pseudo_labeled_dataset_rgb)
        ir_instance_num = len(pseudo_labeled_dataset_ir)
        memory_all.pro_rgb = rgb_instance_num / (rgb_instance_num+ir_instance_num)
        memory_all.pro_ir = ir_instance_num / (rgb_instance_num+ir_instance_num)
        # memory_all.pro_rgb = 0.5
        # memory_all.pro_ir = 0.5

        print("rgb_instance_num: {} ir_instance_num: {}".format(rgb_instance_num,ir_instance_num))

        if epoch < 40:
            train_loader_ir = _stage2_get_train_loader_color(args, dataset_ir, args.height, args.width,
                                            (args.batch_size//2), args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal,train_transformer1=transform_thermal1)

            train_loader_rgb = _stage2_get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            (args.batch_size//2), args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

            train_loader_ir.new_epoch()
            train_loader_rgb.new_epoch()

            trainer.train(args, epoch, train_loader_ir,train_loader_rgb, optimizer,
                        print_freq=args.print_freq, train_iters=len(train_loader_ir), miu=args.miu)
            # del train_loader_ir, train_loader_rgb
        else:
            train_loader = _stage2_get_train_loader_inter(args, dataset_rgb, dataset_ir, args.height, args.width,
                                              (args.batch_size//2), args.workers, args.num_instances, iters,
                                              trainset_rgb=pseudo_labeled_dataset_rgb, trainset_ir=pseudo_labeled_dataset_ir,
                                              no_cam=args.no_cam, train_transformer_rgb=train_transformer_rgb,train_transformer_rgb1=train_transformer_rgb1,
                                              train_transformer_ir=transform_thermal,train_transformer_ir1=transform_thermal1)
        
            train_loader.new_epoch()

            trainer.train(args, epoch, train_loader, optimizer,
                        print_freq=args.print_freq, train_iters=len(train_loader), miu=args.miu)
        

        if ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            mAP = evaluate_sysu_mode(model, args, transform_test, mode='all')

            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(stage2_logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(stage2_logs_dir, 'model_best.pth.tar'))
    _stage2_load_checkpoint_into_model(model, checkpoint, 'stage2 best')
    transform_test = build_test_transform(args)
    print('all')
    evaluate_sysu_mode(model, args, transform_test, mode='all')
#################################

    print('indoor')
    evaluate_sysu_mode(model, args, transform_test, mode='indoor')

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def build_parser(stage=1):
    parser = build_legacy_dataset_parser(
        stage,
        "sysu",
        _WORKING_DIR,
        common_defaults=_PARSER_COMMON_DEFAULTS,
        stage_defaults=_PARSER_STAGE_DEFAULTS,
    )
    return require_dataset_data_dir(parser, "SYSU")


def main(args=None):
    if args is None:
        stage = detect_stage_from_cli()
        args = build_parser(stage=stage).parse_args()
    if not getattr(args, 'data_dir', None):
        raise ValueError('SYSU training requires --data-dir to be provided explicitly.')
    stage = getattr(args, 'stage', None)
    if stage == 1:
        return run_stage1(args)
    if stage == 2:
        return run_stage2(args)
    raise ValueError('Unsupported stage: {0}'.format(stage))


if __name__ == '__main__':
    main()
