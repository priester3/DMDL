import torch
import torch.nn as nn
from torch.nn import init

from .resnet_agw import resnet50 as resnet50_agw


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm)


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
        )
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f_div_c = torch.matmul(theta_x, phi_x) / phi_x.size(-1)

        y = torch.matmul(f_div_c, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        return self.W(y) + x


def weights_init_kaiming(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(module.weight.data, a=0, mode='fan_out')
        init.zeros_(module.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(module.weight.data, 1.0, 0.01)
        init.zeros_(module.bias.data)


class base_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_module, self).__init__()
        self.visible = resnet50_agw(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        return self.visible.maxpool(x)


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()
        model_base = resnet50_agw(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        return self.base.layer4(x)


class embed_net_ori(nn.Module):
    def __init__(self, num_classes=1000, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net_ori, self).__init__()
        self.base_module = base_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        self.gm_pool = gm_pool

        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList([Non_local(256) for _ in range(non_layers[0])])
            self.NL_2 = nn.ModuleList([Non_local(512) for _ in range(non_layers[1])])
            self.NL_3 = nn.ModuleList([Non_local(1024) for _ in range(non_layers[2])])
            self.NL_4 = nn.ModuleList([Non_local(2048) for _ in range(non_layers[3])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])]) or [-1]
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])]) or [-1]
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])]) or [-1]
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])]) or [-1]

        pool_dim = 2048
        self.num_features = pool_dim
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_non_local_block(self, x, layer, modules, indices):
        counter = 0
        for layer_idx in range(len(layer)):
            x = layer[layer_idx](x)
            if layer_idx == indices[counter]:
                x = modules[counter](x)
                counter += 1
        return x

    def forward(self, x1, x2, modal=0, label_1=None, label_2=None):
        single_size = x1.size(0)
        if modal == 0:
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = x1
        else:
            x = x2

        x = self.base_module(x)
        if self.non_local == 'on':
            x = self._forward_non_local_block(x, self.base_resnet.base.layer1, self.NL_1, self.NL_1_idx)
            x = self._forward_non_local_block(x, self.base_resnet.base.layer2, self.NL_2, self.NL_2_idx)
            x = self._forward_non_local_block(x, self.base_resnet.base.layer3, self.NL_3, self.NL_3_idx)
            x = self._forward_non_local_block(x, self.base_resnet.base.layer4, self.NL_4, self.NL_4_idx)
        else:
            x = self.base_resnet(x)

        if self.gm_pool == 'on':
            b, c, _, _ = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1.0 / p)
        else:
            x_pool = self.avgpool(x).view(x.size(0), x.size(1))

        feat = self.bottleneck(x_pool)
        if self.training:
            return feat, feat[:single_size], feat[single_size:], label_1, label_2, x_pool[:single_size], x_pool[single_size:]
        return self.l2norm(feat)


def agw_one(pretrained=False, no_local='on', **kwargs):
    return embed_net_ori(no_local='on', gm_pool='on')
