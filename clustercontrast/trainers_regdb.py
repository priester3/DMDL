from __future__ import print_function, absolute_import

import time

import torch
from torch.nn import functional as F

from .dmdl.acceleration import get_amp_controller, move_to_cuda
from .loss import OriTripletLoss
from .utils.loss import DAN_new
from .utils.meters import AverageMeter


class CausalTrainer_s2(object):
    def __init__(self, encoder, memory=None):
        super(CausalTrainer_s2, self).__init__()
        self.encoder = encoder
        self.memory_all = memory
        self.tri = OriTripletLoss()

    def train(self, args, epoch, data_loader_ir, data_loader_rgb, optimizer,
              print_freq=10, train_iters=400, miu=0.25):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self._runtime_args = args
        amp = get_amp_controller(self, args)

        end = time.time()
        for i in range(train_iters):
            with amp.autocast():
                inputs_ir = data_loader_ir.next()
                inputs_rgb = data_loader_rgb.next()
                data_time.update(time.time() - end)

                inputs_ir, inputs_ir1, labels_ir, clabels_ir, cams_ir, _ = self._parse_data_aug(inputs_ir)
                split_ir = inputs_ir.size(0)
                inputs_ir = torch.cat((inputs_ir, inputs_ir1), 0)
                labels_ir = torch.cat((labels_ir, labels_ir), -1)
                clabels_ir = torch.cat((clabels_ir, clabels_ir), -1)
                cams_ir = torch.cat((cams_ir, cams_ir), -1)

                inputs_rgb, inputs_rgb1, labels_rgb, clabels_rgb, cams_rgb, _ = self._parse_data_aug(inputs_rgb)
                split_rgb = inputs_rgb.size(0)
                inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
                labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
                clabels_rgb = torch.cat((clabels_rgb, clabels_rgb), -1)
                cams_rgb = torch.cat((cams_rgb, cams_rgb), -1)

                _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, _, _ = self._forward(
                    inputs_rgb, inputs_ir, label_1=labels_rgb, label_2=labels_ir, modal=0
                )

                def comb(p1, p2, lam):
                    return (1 - lam) * p1 + lam * p2

                i2r = bool(epoch % 2)
                targets_onehot_noise_ir = F.one_hot(labels_ir, self.memory_all.num_all).float().cuda()
                prob_ir = self.memory_all.get_predict(f_out_ir, i2r).detach()
                targets_corrected_ir = comb(
                    torch.cat((prob_ir[split_ir:], prob_ir[:split_ir]), dim=0),
                    targets_onehot_noise_ir,
                    clabels_ir.unsqueeze(1),
                )
                targets_onehot_noise_rgb = F.one_hot(labels_rgb, self.memory_all.num_all).float().cuda()
                prob_rgb = self.memory_all.get_predict(f_out_rgb, i2r).detach()
                targets_corrected_rgb = comb(
                    torch.cat((prob_rgb[split_rgb:], prob_rgb[:split_rgb]), dim=0),
                    targets_onehot_noise_rgb,
                    clabels_rgb.unsqueeze(1),
                )

                if epoch < 5:
                    targets_corrected_ir = targets_onehot_noise_ir
                    targets_corrected_rgb = targets_onehot_noise_rgb

                loss_ir, loss_ir_y = self.memory_all(f_out_ir, labels_ir, targets_corrected_ir, mode='ir', i2r=i2r)
                loss_rgb, loss_rgb_y = self.memory_all(f_out_rgb, labels_rgb, targets_corrected_rgb, mode='rgb', i2r=i2r)

                loss_tri_ir, _ = self.tri(f_out_ir, labels_ir)
                loss_tri_rgb, _ = self.tri(f_out_rgb, labels_rgb)
                loss_try = DAN_new(f_out_rgb[:split_rgb], f_out_rgb[split_rgb:]) + DAN_new(f_out_ir[:split_ir], f_out_ir[split_ir:])
                loss_cross = loss_ir_y + loss_rgb_y
                loss_tri = loss_tri_ir + loss_tri_rgb

                if epoch > 20:
                    sigma = 1.0
                elif epoch > 10:
                    sigma = 0.5
                else:
                    sigma = 0.0

                loss = loss_ir + loss_rgb + miu * loss_cross + sigma * loss_tri + loss_try

            amp.step(loss, optimizer)

            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(
                    'Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    'Loss ir {:.3f}\t'
                    'Loss rgb {:.3f}\t'
                    'Loss ir y {:.3f}\t'
                    'Loss rgb y {:.3f}\t'
                    'Loss try {:.3f}\t'
                    'Loss tri ir {:.3f}\t'
                    'Loss tri rgb {:.3f}\t'.format(
                        epoch, i + 1, len(data_loader_rgb),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg,
                        loss_ir.item(), loss_rgb.item(),
                        loss_ir_y.item(), loss_rgb_y.item(),
                        loss_try.item(),
                        loss_tri_ir.item(), loss_tri_rgb.item(),
                    )
                )

    def _parse_data_aug(self, inputs):
        imgs, imgs1, _, pids, cpids, cids, indexes = inputs
        args = self._runtime_args
        return (
            move_to_cuda(imgs, args),
            move_to_cuda(imgs1, args),
            move_to_cuda(pids, args),
            move_to_cuda(cpids, args),
            move_to_cuda(cids, args),
            move_to_cuda(indexes, args),
        )

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)


class CausalTrainer_s2_inter(object):
    def __init__(self, encoder, memory=None):
        super(CausalTrainer_s2_inter, self).__init__()
        self.encoder = encoder
        self.memory_all = memory
        self.tri = OriTripletLoss()

    def train(self, args, epoch, data_loader, optimizer,
              print_freq=10, train_iters=400, miu=0.5):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self._runtime_args = args
        amp = get_amp_controller(self, args)

        end = time.time()
        for i in range(train_iters):
            with amp.autocast():
                inputs = data_loader.next()
                data_time.update(time.time() - end)

                inputs_rgb, inputs_rgb1, labels_rgb, clabels_rgb, cams_rgb, \
                inputs_ir, inputs_ir1, labels_ir, clabels_ir, cams_ir = self._parse_data_aug(inputs)

                split_ir = inputs_ir.size(0)
                inputs_ir = torch.cat((inputs_ir, inputs_ir1), 0)
                labels_ir = torch.cat((labels_ir, labels_ir), -1)
                clabels_ir = torch.cat((clabels_ir, clabels_ir), -1)
                cams_ir = torch.cat((cams_ir, cams_ir), -1)

                split_rgb = inputs_rgb.size(0)
                inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
                labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
                clabels_rgb = torch.cat((clabels_rgb, clabels_rgb), -1)
                cams_rgb = torch.cat((cams_rgb, cams_rgb), -1)

                _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, _, _ = self._forward(
                    inputs_rgb, inputs_ir, label_1=labels_rgb, label_2=labels_ir, modal=0
                )

                def comb(p1, p2, lam):
                    return (1 - lam) * p1 + lam * p2

                i2r = bool(epoch % 2)
                targets_onehot_noise_ir = F.one_hot(labels_ir, self.memory_all.num_all).float().cuda()
                prob_ir = self.memory_all.get_predict(f_out_ir, i2r).detach()
                targets_corrected_ir = comb(
                    torch.cat((prob_ir[split_ir:], prob_ir[:split_ir]), dim=0),
                    targets_onehot_noise_ir,
                    clabels_ir.unsqueeze(1),
                )
                targets_onehot_noise_rgb = F.one_hot(labels_rgb, self.memory_all.num_all).float().cuda()
                prob_rgb = self.memory_all.get_predict(f_out_rgb, i2r).detach()
                targets_corrected_rgb = comb(
                    torch.cat((prob_rgb[split_rgb:], prob_rgb[:split_rgb]), dim=0),
                    targets_onehot_noise_rgb,
                    clabels_rgb.unsqueeze(1),
                )

                loss_ir, loss_ir_y = self.memory_all(f_out_ir, labels_ir, targets_corrected_ir, mode='ir', i2r=i2r)
                loss_rgb, loss_rgb_y = self.memory_all(f_out_rgb, labels_rgb, targets_corrected_rgb, mode='rgb', i2r=i2r)

                loss_tri_ir, _ = self.tri(f_out_ir, labels_ir)
                loss_tri_rgb, _ = self.tri(f_out_rgb, labels_rgb)
                loss_tri_all, _ = self.tri(torch.cat((f_out_rgb, f_out_ir), 0), torch.cat((labels_rgb, labels_ir), -1))
                loss_try = DAN_new(f_out_rgb[:split_rgb], f_out_rgb[split_rgb:]) + DAN_new(f_out_ir[:split_ir], f_out_ir[split_ir:])
                loss_cross = loss_ir_y + loss_rgb_y
                loss_tri = loss_tri_ir + loss_tri_rgb + loss_tri_all
                sigma = 1.0
                loss = loss_ir + loss_rgb + miu * loss_cross + sigma * loss_tri + loss_try

            amp.step(loss, optimizer)

            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print(
                    'Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    'Loss ir {:.3f}\t'
                    'Loss rgb {:.3f}\t'
                    'Loss ir y {:.3f}\t'
                    'Loss rgb y {:.3f}\t'
                    'Loss try {:.3f}\t'
                    'Loss tri ir {:.3f}\t'
                    'Loss tri rgb {:.3f}\t'
                    'Loss tri all {:.3f}'.format(
                        epoch, i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg,
                        loss_ir.item(), loss_rgb.item(),
                        loss_ir_y.item(), loss_rgb_y.item(),
                        loss_try.item(),
                        loss_tri_ir.item(), loss_tri_rgb.item(),
                        loss_tri_all.item(),
                    )
                )

    def _parse_data_aug(self, inputs):
        imgs10, imgs11, _, pids1, cpids1, cids1, _, imgs20, imgs21, _, pids2, cpids2, cids2, _ = inputs
        args = self._runtime_args
        return (
            move_to_cuda(imgs10, args),
            move_to_cuda(imgs11, args),
            move_to_cuda(pids1, args),
            move_to_cuda(cpids1, args),
            move_to_cuda(cids1, args),
            move_to_cuda(imgs20, args),
            move_to_cuda(imgs21, args),
            move_to_cuda(pids2, args),
            move_to_cuda(cpids2, args),
            move_to_cuda(cids2, args),
        )

    def _forward(self, x1, x2, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)
