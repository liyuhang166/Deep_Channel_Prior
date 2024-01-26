import math
import torch
import torch.nn as nn


# 自定义channel方向归一化
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor':  # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# padding方式选择
def get_padding(padding, pad_type):
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)   # 镜像填充
    elif pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)  # 重复填充
    elif pad_type == 'zero':
        return nn.ZeroPad2d(padding)         # 零填充
    assert 0, "Unsupported padding type: {}".format(pad_type)


# 归一化层选择
def get_norm(norm_dim, norm_type, **kwargs):
    if norm_type == 'bn':
        return nn.BatchNorm2d(norm_dim)      # batch方向归一化
    elif norm_type == 'in':
        return nn.InstanceNorm2d(norm_dim)   # 一个channel内做归一化
    elif norm_type == 'ln':
        return LayerNorm(norm_dim)           # channel方向归一化
    elif norm_type == 'none':
        return None
    assert 0, "Unsupported normalization: {}".format(norm_type)


# 激活函数选择
def get_activation(activ_type):
    if activ_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activ_type == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activ_type == 'prelu':
        return nn.PReLU()
    elif activ_type == 'selu':
        return nn.SELU(inplace=True)
    elif activ_type == 'tanh':
        return nn.Tanh()
    elif activ_type == 'sigmoid':
        return nn.Sigmoid()
    elif activ_type == 'softmax':
        return nn.Softmax(dim=1)
    elif activ_type == 'none':
        return lambda x: x
    assert 0, "Unsupported activation: {}".format(activ_type)


##################################################################################
# My Block
##################################################################################

# 1*1卷积核
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# 3*3卷积核
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# 自定义卷积核
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0,
                 norm='none', activation='relu', pad_type='zero', dilation=1, bias=True):
        super(Conv2dBlock, self).__init__()
        # initialize padding
        self.pad = get_padding(padding, pad_type)
        # initialize normalization
        self.norm = get_norm(output_dim, norm)
        # initialize activation
        self.activation = get_activation(activation)
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(self.pad(x))  # padding->conv
        if self.norm:
            x = self.norm(x)        # conv->norm
        if self.activation:
            x = self.activation(x)  # norm->activation
        return x
