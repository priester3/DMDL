import collections
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn


class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        return inputs.mm(ctx.features.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.to(grad_outputs.dtype))

        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        return inputs.mm(ctx.features.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, corrected_targets, momentum=None, reduction='mean'):
        inputs = F.normalize(inputs, dim=1).cuda()
        current_momentum = self.momentum if momentum is None else momentum
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, current_momentum)
        else:
            outputs = cm(inputs, targets, self.features, current_momentum)
        outputs /= self.temp
        return F.cross_entropy(outputs, corrected_targets, reduction=reduction)

    def dis_loss(self, inputs):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = inputs.mm(self.features.clone().T)
        outputs /= self.temp
        outputs = nn.Softmax(1)(outputs)
        log_preds = torch.log(outputs + 1e-20)
        return (-(1 / self.num_samples) * log_preds).sum(1).mean()

    def get_predict(self, inputs):
        inputs = F.normalize(inputs, dim=1).cuda()
        outputs = inputs.mm(self.features.clone().T)
        outputs /= self.temp
        return F.softmax(outputs.detach(), dim=1)


class CM_Cam_Cross(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, cams, features, pids, cids, momentum):
        ctx.features = features
        ctx.pids = pids
        ctx.cids = cids
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, cams)
        return inputs.mm(ctx.features.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, cams = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.to(grad_outputs.dtype))

        for i in range(inputs.shape[0]):
            up_mask = (cams[i] == ctx.cids).float() * (targets[i] == ctx.pids).float()
            up_idx = torch.nonzero(up_mask > 0).squeeze(-1)
            if len(up_idx) == 0:
                continue
            ctx.features[up_idx] = ctx.momentum * ctx.features[up_idx] + (1.0 - ctx.momentum) * inputs[i]
            ctx.features[up_idx] /= ctx.features[up_idx].norm()

        return grad_inputs, None, None, None, None, None, None


def cm_cam_cross(inputs, indexes, cams, features, pids, cids, momentum=0.2):
    return CM_Cam_Cross.apply(inputs, indexes, cams, features, pids, cids, torch.Tensor([momentum]).to(inputs.device))


class CamMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(CamMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.register_buffer('proxy', torch.zeros(num_samples, num_features))
        self.register_buffer('pids', torch.zeros(num_samples).long())
        self.register_buffer('cids', torch.zeros(num_samples).long())

    def forward(self, inputs, targets, cams):
        inputs = F.normalize(inputs, dim=1).cuda()
        sims = cm_cam_cross(inputs, targets, cams, self.proxy, self.pids, self.cids)
        sims /= self.temp

        loss = torch.tensor([0.0]).cuda()
        for i in range(inputs.shape[0]):
            mask = (cams[i] == self.cids).float()
            idx = torch.nonzero(mask > 0).squeeze(-1)
            if len(idx) == 0:
                continue
            loss += F.cross_entropy(sims[i, idx], targets[i])

        return loss / inputs.shape[0]
