import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
import os
import random
import numpy as np
import math 

#这个是修改了resnet模块之后的代码，不能动 
## Adapted from https://github.com/joaomonteirof/e2e_antispoofing
## https://github.com/yzyouzhang/AIR-ASVspoof
## https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/newfunctions/

class ChannelAttention(nn.Module):  # Channel Attention Module
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

        
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        # self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0) == 1:
            attentions = F.softmax(torch.tanh(weights), dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()), dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5 * torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted + noise).std(1)

            representations = torch.cat((avg_repr, std_repr), 1)

            return representations

        ##################################################################
#修改前的EMA模块      
# class EMA(nn.Module):
#     def __init__(self,channels,factor=8):
#         super(EMA,self).__init__()
#         self.groups = factor
#         assert channels//self.groups>0
#         self.softmax = nn.Softmax(-1)
#         self.agp = nn.AdaptiveAvgPool2d((1,1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None,1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1,None))
#         self.gn = nn.GroupNorm(channels//self.groups,channels//self.groups)
#         self.conv1 =nn.Conv2d(channels//self.groups,channels//self.groups,kernel_size=1,stride=1,padding=0)
#         self.conv2 = nn.Conv2d(channels//self.groups,channels//self.groups,kernel_size=3,stride=1,padding=1)

#     def forward(self,x):
#         b,c,h,w = x.size()
#         group_x = x.reshape(b*self.groups,-1,h,w)
#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0,1,3,2)
#         hw = self.conv1(torch.cat([x_h,x_w],dim=2))
#         x_h,x_w = torch.split(hw,[h,w],dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv2(group_x)
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x*weights.sigmoid()).reshape(b,c,h,w)

#加了噪声之后的EMA
class EMA(nn.Module):
    def __init__(self, channels, factor=8, noise_factor=0.1):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.noise_factor = noise_factor  # 控制噪声的强度
        self.attention=SelfAttention(256)

    def add_noise(self, tensor):
        # 添加高斯噪声
        noise = torch.randn_like(tensor) * self.noise_factor
        return tensor + noise

    def forward(self, x):
        b, c, h, w = x.size()
        # 在输入特征上添加噪声
        x = self.add_noise(x)
        
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv2(group_x)
        
        # 在卷积操作后添加噪声
        x2 = self.add_noise(x2)
        x11 = (x1 + x2) / 2
        return x11.reshape(b,c,h,w)
        
        # x = self.activation(self.bn5(x11)).squeeze(2)
        # stats = self.attention(x.permute(0, 2, 1).contiguous())
       

    ####################################################################################

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.channel = ChannelAttention(self.expansion * planes)  # Channel Attention Module
        self.spatial = SpatialAttention()  # Spatial Attention Module

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
        
        

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        CBAM_Cout = self.channel(out)
        out = out * CBAM_Cout
        CBAM_Sout = self.spatial(out)
        out = out * CBAM_Sout
        out += shortcut
        return out

    # def forward(self, x):
    #     out = F.relu(self.bn1(x))
    #     shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
    #     out = self.conv1(out)
    #     out = self.conv2(F.relu(self.bn2(out)))
    #     # shortcut = x
    #     # out = self.conv1(x)
    #     # out=self.bn1(out)
    #     # out=F.relu(out)
    #     # out = F.relu(self.bn1(out))
    #     # out = self.conv2(out)
    #     CBAM_Cout = self.channel(out)
    #     out = out * CBAM_Cout
    #     CBAM_Sout = self.spatial(out)
    #     out = out * CBAM_Sout
    #     out += shortcut
    #     return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock]}


def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = cudnn_deterministic

class ResNet(nn.Module):
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2):
        self.in_planes = 16
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.LeakyReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2, enc_dim)
        # self.fc = nn.Linear(520, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.initialize_params()
        # self.attention = SelfAttentionWithLSH(256)
        self.attention = SelfAttention(256)
        self.SEC = EMA(256)


    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)
        

    def forward(self, x):
        # 原始的
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.SEC(x)
        x = self.activation(self.bn5(x)).squeeze(2)
        # print(x.shape)
        stats = self.attention(x.permute(0, 2, 1).contiguous())
        # print(stats.shape)
        feat = self.fc(stats)
        mu = self.fc_mu(feat)
        return feat, mu
    
        #改了的EMA
        # x = self.conv1(x)
        # x = self.activation(self.bn1(x))
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.conv5(x)
        # x = self.activation(self.bn5(x))
        # print(x.shape)
        # stats = self.SEC(x).contiguous()
        # # print(stats.shape)
        # feat = self.fc(stats)
        # mu = self.fc_mu(feat)
        # return feat, mu
    
        #加了噪声的EMA    
        # x = self.conv1(x)
        # x = self.activation(self.bn1(x))
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.conv5(x)
        # x = self.activation(self.bn5(x))
        # x = self.SEC(x)
        # # print(x.shape)
        # x = x.squeeze(2).permute(0, 2, 1).contiguous()
        # # print(x.shape)
        # state = self.attention(x)
        # feat = self.fc(state)
        # mu = self.fc_mu(feat)
        # return feat, mu
        
        


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class TypeClassifier(nn.Module):
    def __init__(self, enc_dim, nclasses, lambda_=0.05, ADV=True):
        super(TypeClassifier, self).__init__()
        self.adv = ADV
        if self.adv:
            self.grl = GradientReversal(lambda_)
        self.classifier = nn.Sequential(nn.Linear(enc_dim, enc_dim // 2),
                                        nn.Dropout(0.3),
                                        nn.ReLU(),
                                        nn.Linear(enc_dim // 2, nclasses),
                                        nn.ReLU())

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        if self.adv:
            x = self.grl(x)
        return self.classifier(x)

    
if __name__=='__main__':
    model=EMA(256).to('cuda')
    input = torch.rand(32,256,1,94).to('cuda')
    output = model(input)
    print(output.shape)