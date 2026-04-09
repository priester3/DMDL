from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, pids, momentum):
        ctx.features = features
        ctx.pids = pids
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.to(grad_outputs.dtype))

        # momentum update
        for x, y in zip(inputs, targets):
            up_mask = (y == ctx.pids).float()
            up_idx = torch.nonzero(up_mask > 0).squeeze(-1)
            ctx.features[up_idx] = ctx.momentum * ctx.features[up_idx] + (1. - ctx.momentum) * x
            ctx.features[up_idx] /= ctx.features[up_idx].norm()

        return grad_inputs, None, None, None, None


def cm(inputs, indexes, features, pids, momentum=0.5):
    return CM.apply(inputs, indexes, features, pids, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples_rgb, num_samples_ir, num_all, temp=0.05, momentum=0.2, pro_rgb=None, pro_ir=None):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples_rgb = num_samples_rgb
        self.num_samples_ir = num_samples_ir
        self.num_all = num_all

        self.momentum = momentum
        self.temp = temp
        self.pro_rgb = pro_rgb
        self.pro_ir = pro_ir

        self.register_buffer('features_rgb', torch.zeros(num_samples_rgb, num_features))
        self.register_buffer('features_ir', torch.zeros(num_samples_ir, num_features))
        self.register_buffer('features_all', torch.zeros(num_all, num_features))
        self.register_buffer('pids_rgb', torch.zeros(num_samples_rgb).long())
        self.register_buffer('pids_ir', torch.zeros(num_samples_ir).long())
    
    def forward(self, inputs, targets, corrected_targets, mode='rgb', i2r=None, reduction='mean'):
        inputs = F.normalize(inputs, dim=1)
        
        if mode == 'rgb':
            logits_rgb = cm(inputs, targets, self.features_rgb, self.pids_rgb, self.momentum)
            logits_rgb /= self.temp
            if not i2r:
                logits_ir = cm(inputs, targets, self.features_ir, self.pids_ir, self.momentum)
            else:
                logits_ir = torch.mm(inputs, self.features_ir.clone().T)
            logits_ir /= self.temp
            loss_yc = F.cross_entropy(logits_rgb, corrected_targets[:,self.pids_rgb], reduction=reduction)
        else:
            if i2r:
                logits_rgb = cm(inputs, targets, self.features_rgb, self.pids_rgb, self.momentum)
            else:
                logits_rgb = torch.mm(inputs, self.features_rgb.clone().T)
            logits_rgb /= self.temp
            logits_ir = cm(inputs, targets, self.features_ir, self.pids_ir, self.momentum)
            logits_ir /= self.temp
            loss_yc = F.cross_entropy(logits_ir, corrected_targets[:,self.pids_ir], reduction=reduction)
        logits_rgb = F.softmax(logits_rgb, dim=1)
        logits_ir = F.softmax(logits_ir, dim=1)

        logits_rgb_all = torch.zeros(inputs.size(0), self.num_all, dtype=logits_rgb.dtype, device=inputs.device)
        logits_ir_all = torch.zeros(inputs.size(0), self.num_all, dtype=logits_ir.dtype, device=inputs.device)
        logits_rgb_all[:,self.pids_rgb] = logits_rgb
        logits_ir_all[:,self.pids_ir] = logits_ir

        logits_all = logits_rgb_all*self.pro_rgb + logits_ir_all*self.pro_ir
        
        loss_y = -torch.sum(corrected_targets * torch.log(logits_all), dim=1).mean()

        if i2r:
            self.update_feature_all(inputs, targets, self.pids_rgb, corrected_targets) 
        else:
            self.update_feature_all(inputs, targets, self.pids_ir, corrected_targets) 

        return loss_yc, loss_y
    
    def update_feature_all(self, inputs, targets, pids, corrected_targets):
        for x, y, y_c in zip(inputs, targets, corrected_targets):
            up_mask = (y == pids).float()
            up_idx = torch.nonzero(up_mask > 0).squeeze(-1)
            m_new =  self.momentum / (max(y_c[y], self.momentum))
            self.features_all[up_idx] = m_new * self.features_all[up_idx] + (1. - m_new) * x
            self.features_all[up_idx] /= self.features_all[up_idx].norm()

    
    def dis_loss(self, inputs, mode='rgb'):
        inputs = F.normalize(inputs, dim=1)
        if mode == 'rgb':
            outputs = inputs.mm(self.features_rgb.clone().T )
            Num_all = self.num_samples_rgb
        elif mode == 'ir':
            outputs = inputs.mm(self.features_ir.clone().T )
            Num_all = self.num_samples_ir
        outputs /= self.temp
        outputs = F.softmax(outputs, dim=1)
        log_preds = torch.log(outputs + 1e-20)

        loss = (- (1 / Num_all) * log_preds).sum(1).mean()
        return loss
    
    def get_predict(self, inputs, i2r=None):
        inputs = F.normalize(inputs, dim=1)
        outputs = inputs.mm(self.features_all.clone().T )
        outputs /= self.temp
        outputs = F.softmax(outputs.detach(), dim=1)

        return outputs
