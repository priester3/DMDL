import torch
import torch.nn as nn
from torch.autograd import Variable

# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res


def lossb_expect(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).cuda()
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda()
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]

        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss


def weight_learner(cfeatures, pre_features, pre_weight1, global_epoch=0, iter=0):
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    lrbl = 1.0
    optimizerbl = torch.optim.SGD([weight], lr=lrbl, momentum=0.9)

    for epoch in range(20):
        lr = lrbl * (0.1 ** (epoch // 10))
        for param_group in optimizerbl.param_groups:
            param_group['lr'] = lr
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = lossb_expect(all_feature, softmax(all_weight), 1)
        lossp = softmax(weight).pow(2).sum()
        lambdap = 70.0 * max((1 ** (global_epoch // 5)),
                                     0.01)
        lossg = lossb / lambdap + lossp

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * 0.9 + cfeatures * (
                    1 - 0.9)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * 0.9 + weight * (
                    1 - 0.9)

    else:
        pre_features = pre_features * 0.9 + cfeatures * (1 - 0.9)
        pre_weight1 = pre_weight1 * 0.9 + weight * (1 - 0.9)

    softmax_weight = softmax(weight)

    return softmax_weight, pre_features, pre_weight1