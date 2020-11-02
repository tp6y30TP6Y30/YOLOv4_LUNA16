#!/usr/bin/python3
#coding=utf-8
"""
Inspired by https://github.com/lalonderodney/SegCaps/blob/master/capsule_layers.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.bernoulli as bernoulli
import numpy as np

from layers import CReLU


__all__ = ['CapToLength', 'FlatCap', 'ConvertToCapsule', 'NormalizeCapsule', 'ConvCapsuleLayer',
           'DeconvCapsuleLayer', 'ConvCapResidualBlock', 'RFB', 'DropoutCap', 'DenseBlock']

class CapToLength(nn.Module):
    """ Convert a capsule output to usual tensor
    """
    def __init__(self, keepdim=False):
        super(CapToLength, self).__init__()
        self.keepdim = keepdim

    def forward(self, x):
        """
        x: (batch_size, d, h, w, num_cap, num_atom)
        :return: (batch_size, d, h, w, num_cap)
        """
        out = torch.norm(x, p=2, dim=-1, keepdim=self.keepdim)
        return out


class FlatCap(nn.Module):
    """
    Flattening a capsule in channel dimension
    """
    def __init__(self):
        super(FlatCap, self).__init__()

    def forward(self, x):
        """
        x: (batch_size, d, h, w, num_cap, num_atom)
        :return: (batch_size, num_cap*dim_cap, d, h, w)
        """
        assert len(x.size()) == 6, 'The input is not a capsule.'
        batch_size, d, h, w, num_cap, num_atom = x.size()
        out = x.view(batch_size, -1, num_cap*num_atom)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_cap*num_atom, d, h, w)
        return out


class ConvertToCapsule(nn.Module):
    """
    Convert a tensor to a capsule
    """
    def __init__(self, num_cap=1):
        super(ConvertToCapsule, self).__init__()
        self.num_cap = num_cap

    def forward(self, x):
        """
        x: (batch, channel, d, h, w)
        :return: (batch, d, h, w, num_cap, num_atom)
        """
        assert len(x.size()) == 5, 'The input is not a 3-D NCDHW tensor.'
        batch_size, channel, d, h, w = x.size()
        assert channel % self.num_cap == 0, 'The num_atom is not a integer.'
        num_atom = int(channel / self.num_cap)
        out = x.view(batch_size, channel, -1)
        out = out.transpose(1, 2).contiguous().view(batch_size, d, h, w, self.num_cap, num_atom)
        return out


class NormalizeCapsule(nn.Module):
    """
    Normalization a capsule in channel dimension
    """
    def __init__(self, method='instancenorm', inplace=False):
        super(NormalizeCapsule, self).__init__()
        self.method = method
        self.inplace = inplace
        self.flapcap = FlatCap()

    def forward(self, x):
        """
        x: (batch_size, d, h, w, num_cap, num_atom)
        :return: (batch_size, d, h, w, num_cap, num_atom)
        """
        assert x.dim() == 6, 'The input is not a 3-D capsule.'

        # Normalize only in train mode
        if not self.train:
            return x

        if self.inplace:
            out = x
        else:
            out = x.clone()

        batch_size, d, h, w, num_cap, num_atom = x.size()
        normalization = nn.ModuleDict({'batchnorm': nn.BatchNorm3d(num_cap * num_atom),
                                       'instancenorm': nn.InstanceNorm3d(num_cap * num_atom),
                                       'groupnorm': nn.GroupNorm(num_cap, num_cap * num_atom),
                                       'none': nn.Sequential()})
        norm = normalization[self.method]

        out = self.flapcap(out)
        out = norm(out)
        out = out.view(batch_size, num_cap * num_atom, -1)
        out = out.transpose(1, 2).contiguous().view(batch_size, d, h, w, num_cap, num_atom)
        return out


class ConvCapsuleLayer(nn.Module):
    def __init__(self, input_num_atoms, num_capsule, num_atoms, kernel_size,
                 strides=1, padding=0, dilation=1, groups=1, routings=3, output_uji=False):
        super(ConvCapsuleLayer, self).__init__()
        use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.input_num_atoms = input_num_atoms
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.routings = routings
        self.output_uji = output_uji
        self.conv_ = nn.Conv3d(self.input_num_atoms, self.num_capsule * self.num_atoms, kernel_size=kernel_size,
                               stride=strides, padding=padding, dilation=dilation, groups=groups, bias=False)
        nn.init.kaiming_normal_(self.conv_.weight)
        self.b = nn.Parameter(torch.FloatTensor([0.1]).repeat(1, 1, 1, self.num_capsule, self.num_atoms))
        self.act_fun = self._squash

    @staticmethod
    def _squash(input_tensor):
        norm = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

    @staticmethod
    def _squash2(input_tensor):
        """
        From 'Capsule Network Performance on Complex Data'
        https://arxiv.org/pdf/1712.03480.pdf
        """
        norm = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        norm_exp = torch.exp(norm)
        out = (input_tensor / norm) * (1 - 1 / norm_exp)
        return out

    def forward(self, input):
        """
        input: [batch_size, input_depth, input_height, input_width, input_num_capsule, input_num_atoms]
        return: activation, output
        """
        assert len(input.shape) == 6, 'The input tensor is not a 3-D capsule, which has 6 dimensions.'
        batch_size, input_depth, input_height, input_width, input_num_capsule, input_num_atoms = input.shape
        input_trans = input.permute(4, 0, 5, 1, 2, 3).contiguous()
        input_trans = input_trans.view(batch_size * input_num_capsule, input_num_atoms, input_depth,
                                       input_height, input_width)
        output = self.conv_(input_trans)  # (batch_size*input_num_capsule, num_capsule*num_atoms, d, h, w)
        _, _, conv_depth, conv_height, conv_width = output.shape
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        output = output.view(batch_size, input_num_capsule, conv_depth, conv_height, conv_width,
                             self.num_capsule, self.num_atoms)
        votes = output.detach()

        votes_t_shape = [6, 0, 1, 2, 3, 4, 5]
        r_t_shape = [1, 2, 3, 4, 5, 6, 0]
        logit_shape = [1, batch_size, input_num_capsule, conv_depth, conv_height, conv_width, self.num_capsule]

        votes_trans = votes.permute(votes_t_shape)
        logits = torch.empty(logit_shape, device=self.device).fill_(0.0)
        biases_replicated = self.b.repeat(conv_depth, conv_height, conv_width, 1, 1).detach()

        for _ in range(self.routings - 1):
            route = F.softmax(logits, dim=-1)
            preactivate_unrolled = route * votes_trans
            preact_trans = preactivate_unrolled.permute(r_t_shape)
            preactivate = preact_trans.sum(dim=1) + biases_replicated
            activation = self.act_fun(preactivate)
            act_3d = torch.unsqueeze(activation, 1)
            tile_shape = np.ones(len(votes_t_shape), dtype=np.int32).tolist()
            tile_shape[1] = input_num_capsule
            act_replicated = act_3d.repeat(tile_shape)
            distances = (votes * act_replicated).sum(dim=-1)
            logits += distances

        # Save GPU memory
        if self.routings > 1:
            del votes, route, votes_trans, preactivate_unrolled, preact_trans, biases_replicated
            del preactivate, activation, act_3d, act_replicated, distances

        # The last iteration is done on the original output, without the routing weights update
        biases_replicated = self.b.repeat(conv_depth, conv_height, conv_width, 1, 1)
        output = output.permute(votes_t_shape) # [num_atom, batch_size, input_num_cap, d, h, w, num_cap]
        route = F.softmax(logits, dim=-1)
        preactivate_unrolled = route * output
        preact_trans = preactivate_unrolled.permute(r_t_shape)
        preactivate = preact_trans.sum(dim=1) + biases_replicated  # sum at input_num_cap's dimension
        activation = self.act_fun(preactivate)
        if self.output_uji: # Fast CapsNet https://arxiv.org/pdf/1806.07416.pdf
            uji = torch.sum(output, 2, keepdim=True)  # sum at input_num_cap's dimension
            uji = uji.permute(r_t_shape).contiguous().squeeze(1)  # (batch_size, d, h, w, num_cap, num_atom)
            return activation, uji
        else:
            return activation


class DeconvCapsuleLayer(nn.Module):
    def __init__(self, input_num_atoms, num_capsule, num_atoms, kernel_size,
                 strides, padding=0, groups=1, routings=3, output_uji=False):
        super(DeconvCapsuleLayer, self).__init__()
        use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.input_num_atoms = input_num_atoms
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.routings = routings
        self.output_uji = output_uji
        self.deconv_ = nn.ConvTranspose3d(self.input_num_atoms, self.num_capsule * self.num_atoms,
                                          kernel_size=self.kernel_size, stride=self.strides, padding=self.padding,
                                          groups=self.groups, bias=False)
        nn.init.kaiming_normal_(self.deconv_.weight)
        self.b = nn.Parameter(torch.FloatTensor([0.1]).repeat(1, 1, 1, self.num_capsule, self.num_atoms))
        self.act_fun = self._squash

    @staticmethod
    def _squash(input_tensor):
        norm = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

    @staticmethod
    def _squash2(input_tensor):
        norm = torch.norm(input_tensor, p=2, dim=-1, keepdim=True)
        norm_exp = torch.exp(norm)
        out = (input_tensor / norm) * (1 - 1 / norm_exp)
        return out

    def forward(self, input):
        """
        input: [batch_size, input_depth, input_height, input_width, input_num_capsule, input_num_atoms]
        return: activation
        """
        assert len(input.shape) == 6, 'The input tensor must have 6 dimensions.'
        batch_size, input_depth, input_height, input_width, input_num_capsule, input_num_atoms = input.shape
        input_trans = input.permute(4, 0, 5, 1, 2, 3).contiguous()
        input_trans = input_trans.view(batch_size * input_num_capsule, input_num_atoms, input_depth, input_height, input_width)
        output = self.deconv_(input_trans)  # (batch_size*input_num_capsule, num_capsule*num_atoms, d, h, w)
        _, _, conv_depth, conv_height, conv_width = output.shape
        output = output.permute(0, 2, 3, 4, 1).contiguous()
        output = output.view(batch_size, input_num_capsule, conv_depth, conv_height, conv_width, self.num_capsule, self.num_atoms)
        votes = output.detach()

        votes_t_shape = [6, 0, 1, 2, 3, 4, 5]
        r_t_shape = [1, 2, 3, 4, 5, 6, 0]
        logit_shape = [1, batch_size, input_num_capsule, conv_depth, conv_height, conv_width, self.num_capsule]

        votes_trans = votes.permute(votes_t_shape)
        logits = torch.empty(logit_shape, device=self.device).fill_(0.0)
        biases_replicated = self.b.repeat(conv_depth, conv_height, conv_width, 1, 1).detach()

        for _ in range(self.routings - 1):
            route = F.softmax(logits, dim=-1)
            preactivate_unrolled = route * votes_trans
            preact_trans = preactivate_unrolled.permute(r_t_shape)
            preactivate = preact_trans.sum(dim=1) + biases_replicated
            activation = self.act_fun(preactivate)
            act_3d = torch.unsqueeze(activation, 1)
            tile_shape = np.ones(len(votes_t_shape), dtype=np.int32).tolist()
            tile_shape[1] = input_num_capsule
            act_replicated = act_3d.repeat(tile_shape)
            distances = (votes * act_replicated).sum(dim=-1)
            logits += distances

        # Save GPU memory
        if self.routings > 1:
            del votes, route, votes_trans, preactivate_unrolled, preact_trans, biases_replicated
            del preactivate, activation, act_3d, act_replicated, distances

        # The last iteration is done on the original output, without the routing weights update
        biases_replicated = self.b.repeat(conv_depth, conv_height, conv_width, 1, 1)
        output = output.permute(votes_t_shape) # [num_atom, batch_size, input_num_cap, d, h, w, num_cap]
        route = F.softmax(logits, dim=-1)
        preactivate_unrolled = route * output
        preact_trans = preactivate_unrolled.permute(r_t_shape)
        preactivate = preact_trans.sum(dim=1) + biases_replicated
        activation = self.act_fun(preactivate)

        if self.output_uji: # Fast CapsNet https://arxiv.org/pdf/1806.07416.pdf
            uji = torch.sum(output, 2, keepdim=True)  # sum at input_num_cap's dimension
            uji = uji.permute(r_t_shape).contiguous().squeeze(1)  # (batch_size, d, h, w, num_cap, num_atom)
            return activation, uji
        else:
            return activation


class ConvCapResidualBlock(nn.Module):
    def __init__(self, input_num_cap, input_num_atom, num_cap, num_atom, kernel_size,
                 stride, padding, dilation, routing, norm='instancenorm', relu='relu', debug=False):
        super(ConvCapResidualBlock, self).__init__()
        self.debug = debug
        self.input_num_cap = input_num_cap
        self.input_num_atom = input_num_atom
        self.num_cap = num_cap
        self.num_atom = num_atom

        self.flatcap = FlatCap()
        normalization = nn.ModuleDict({'batchnorm': nn.BatchNorm3d(input_num_cap * input_num_atom),
                                       'instancenorm': nn.InstanceNorm3d(input_num_cap * input_num_atom),
                                       'groupnorm': nn.GroupNorm(input_num_cap, input_num_cap * input_num_atom),
                                       'none': nn.Sequential()})
        self.norm = normalization[norm]
        activation = nn.ModuleDict(
            {'relu': nn.ReLU(),
             'crelu': CReLU(),
             'relu6': nn.ReLU6(),
             'lrelu': nn.LeakyReLU(),
             'selu': nn.SELU(),
             'none': nn.Sequential()}
        )
        self.relu = activation[relu]

        # CReLU will double the output channels.
        if relu == 'crelu':
            self.convert_to_cap_in = ConvertToCapsule(num_cap=2*input_num_cap)
        else:
            self.convert_to_cap_in = ConvertToCapsule(num_cap=input_num_cap)
        self.convert_to_cap_out = ConvertToCapsule(num_cap=num_cap)

        self.cap = ConvCapsuleLayer(input_num_atoms=input_num_atom, num_capsule=num_cap, num_atoms=num_atom,
                                    kernel_size=kernel_size, strides=stride, padding=padding, dilation=dilation,
                                    routings=routing)

        # conv3d in shortcut is only needed when stride not 1 or in-channel not equal to out-channel
        if (stride != 1) or (input_num_cap != num_cap) or (input_num_atom != num_atom):
            self.shortcut = nn.Conv3d(input_num_cap*input_num_atom, num_cap*num_atom, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        """
        A convoluted capsule with residual structure
        Design inspiraed by Kaiming's 'Identity Mappings in Deep Residual Networks'
        https://arxiv.org/pdf/1603.05027v3.pdf
        :param x: (n, d, h, w, input_num_cap, input_num_atom)
        :return: (n, d', h', w', num_cap, num_atom) please note its length > 1
        """
        if self.debug: print(f'ConvCapResidualBlock: x={x.size()}')

        out = self.norm(self.flatcap(x))
        out = self.relu(out)
        if self.debug: print(f'ConvCapResidualBlock: relu={out.size()}')

        out = self.cap(self.convert_to_cap_in(out))
        if self.debug: print(f'ConvCapResidualBlock: cap={out.size()}')

        x = self.convert_to_cap_out(self.shortcut(self.flatcap(x)))
        if self.debug: print(f'ConvCapResidualBlock: shortcut={x.size()}')

        out = out + x
        if self.debug: print(f'ConvCapResidualBlock: out+shortcut={out.size()}')

        return out


class RFB(nn.Module):
    """
    https://github.com/ruinmessi/RFBNet
    ECCV 2018, Receptive Field Block Net for Accurate and Fast Object Detection
    Liu, Songtao and Huang, Di and Wang, and Yunhong
    """
    def __init__(self, input_num_cap, input_num_atom, num_cap, num_atom, 
                 stride=1, scale=0.1, visual=1, norm='instancenorm', 
                 debug=False):
        super(RFB, self).__init__()
        self.debug = debug
        self.input_num_cap = input_num_cap
        self.input_num_atom = input_num_atom
        self.num_cap = num_cap
        self.num_atom = num_atom
        self.stride = stride
        self.scale = scale
        self.visual = visual
        inter_num_cap = input_num_cap

        self.flatcap = FlatCap()
        self.convert_to_cap_in = ConvertToCapsule(num_cap=input_num_cap)
        self.convert_to_cap_out = ConvertToCapsule(num_cap=num_cap)
        normalization = nn.ModuleDict({'batchnorm': nn.BatchNorm3d(input_num_cap * input_num_atom),
                                       'instancenorm': nn.InstanceNorm3d(input_num_cap * input_num_atom),
                                       'groupnorm': nn.GroupNorm(input_num_cap, input_num_cap * input_num_atom),
                                       'none': nn.Sequential()})
        self.norm = normalization[norm]

        self.branch0 = nn.Sequential(
            ConvCapsuleLayer(input_num_atoms=input_num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=1, strides=stride,
                             padding=0, dilation=1, groups=1, routings=3),
            ConvCapsuleLayer(input_num_atoms=num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=3, strides=1, padding=visual, dilation=visual, groups=1, routings=3)
        )
        self.branch1 = nn.Sequential(
            ConvCapsuleLayer(input_num_atoms=input_num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=1, strides=1, padding=0, dilation=1, groups=1, routings=3),
            ConvCapsuleLayer(input_num_atoms=num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=3, strides=stride, padding=1, dilation=visual, groups=1, routings=3),
            ConvCapsuleLayer(input_num_atoms=num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=3, strides=1, padding=visual+1, dilation=visual+1, groups=1, routings=3)
        )
        self.branch2 = nn.Sequential(
            ConvCapsuleLayer(input_num_atoms=input_num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=1, strides=1, padding=0, dilation=1, groups=1, routings=3),
            ConvCapsuleLayer(input_num_atoms=num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=3, strides=1, padding=1, dilation=1, groups=1, routings=3),
            ConvCapsuleLayer(input_num_atoms=num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=3, strides=stride, padding=1, dilation=1, groups=1, routings=3),
            ConvCapsuleLayer(input_num_atoms=num_atom, num_capsule=inter_num_cap, num_atoms=num_atom,
                             kernel_size=3, strides=1, padding=2*visual+1, dilation=2*visual+1, groups=1, routings=3)
        )

        self.ConvLinear = ConvCapsuleLayer(input_num_atoms=num_atom, num_capsule=num_cap, num_atoms=num_atom,
                                           kernel_size=1, strides=1, padding=0, dilation=1, groups=1, routings=1)
        self.shortcut = nn.Sequential()
        # conv3d in shortcut is only needed when stride not 1 or in-channel not equal to out-channel
        if (stride != 1) or (input_num_cap * input_num_atom != num_cap * num_atom):
            self.shortcut = nn.Conv3d(input_num_cap * input_num_atom, num_cap * num_atom, kernel_size=1,
                                      stride=stride, padding=0, bias=False)

    def forward(self, x):
        if self.debug: print(f'RFB: x={x.size()}')

        out = F.relu(self.norm(self.flatcap(x)))
        out = self.convert_to_cap_in(out)
        if self.debug: print(f'RFB: out={out.size()}')

        x0 = self.branch0(out)
        x1 = self.branch1(out)
        x2 = self.branch2(out)
        if self.debug: print(f'RFB: x0={x0.size()}')
        if self.debug: print(f'RFB: x1={x1.size()}')
        if self.debug: print(f'RFB: x2={x2.size()}')

        out = torch.cat((x0, x1, x2), 4)
        if self.debug: print(f'RFB: x0+x1+x2={out.size()}')

        out = self.ConvLinear(out)
        if self.debug: print(f'RFB: ConvLinear(out)={out.size()}')

        short = self.shortcut(self.flatcap(x))
        if self.debug: print(f'RFB: shortcut={short.size()}')

        out = self.scale * self.flatcap(out) + short
        if self.debug: print(f'RFB: out+shortcut={out.size()}')

        out = self.convert_to_cap_out(out)
        if self.debug: print(f'RFB: out={out.size()}')

        return out


class DropoutCap(nn.Module):
    """
    Randomly zeroes whole capsules of the input tensor.
    """
    def __init__(self, p=0.5, inplace=False):
        super(DropoutCap, self).__init__()
        use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.drop_possibility = p
        self.keep_possibility = 1 - p
        self.inplace = inplace

    def forward(self, x):
        """
        Use Bernoulli distribution to zero-out on (batch, d, h, w, n_cap) dimensions
        :param x: (batch_size, d, h, w, n_cap, n_atom)
        :return: (batch_size, d, h, w, n_cap, n_atom)  the same as input tensor
        """
        assert x.dim() == 6, 'The input is not a 3-D capsule.'

        # Dropout only in train mode and p>0
        if self.drop_possibility == 0 or not self.train:
            return x

        if self.inplace:
            output = x
        else:
            output = x.clone()

        batch_size, d, h, w, num_cap, num_atom = x.size()
        mask_shape = [batch_size, d, h, w, num_cap]
        mask = torch.empty(mask_shape, device=self.device).fill_(self.keep_possibility)

        if self.drop_possibility == 1:
            mask.fill_(0)
        else:
            mask = bernoulli.Bernoulli(mask).sample() / self.keep_possibility
        mask = mask.unsqueeze(5)
        mask = mask.repeat(1, 1, 1, 1, 1, num_atom)
        output.mul_(mask)

        return output


class _DenseLayer(nn.Sequential):
    def __init__(self, input_num_cap, input_num_atom, growth_rate, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('flatcap', FlatCap())
        self.add_module('norm', nn.InstanceNorm3d(input_num_cap * input_num_atom))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv_to_cap_in', ConvertToCapsule(num_cap=input_num_cap))
        self.add_module('convcap', ConvCapsuleLayer(input_num_atom, growth_rate, input_num_atom, kernel_size=3,
                                                    strides=1, padding=1, dilation=1, groups=1, routings=3))
        self.add_module('drop', DropoutCap(drop_rate, inplace=True))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 4)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_num_cap, input_num_atom, growth_rate, drop_rate=0):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_num_cap + i * growth_rate, input_num_atom, growth_rate, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _DualPathLayer(nn.Module):
    def __init__(self, input_num_cap, input_num_atom, num_cap, num_atom, growth_rate, stride, first_layer):
        super(_DualPathLayer, self).__init__()
        self.input_num_cap = input_num_cap
        self.input_num_atom = input_num_atom
        self.num_cap = num_cap
        self.num_atom = num_atom
        self.growth_rate = growth_rate

        self.norm = NormalizeCapsule('instancenorm', inplace=False)
        self.convcap = ConvCapsuleLayer(input_num_atom, num_cap+growth_rate, num_atom, kernel_size=3,
                                        strides=stride, padding=1, dilation=1, groups=1, routings=3)
        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                FlatCap(),
                nn.Conv3d(input_num_cap*input_num_atom, (num_cap+growth_rate)*num_atom, kernel_size=1, stride=stride, bias=False),
                ConvertToCapsule(num_cap+growth_rate)
            )
        self.debug = False

    def forward(self, x):
        if self.debug: print(f'x={x.size()}')
        if self.debug: print(f'in_cap={self.input_num_cap} '
                             f'in_atom={self.input_num_atom} '
                             f'out_cap={self.num_cap} '
                             f'out_atom={self.num_atom}')
        if self.debug: print(f'growth_rate={self.growth_rate}')

        out = F.relu(self.norm(x))
        if self.debug: print(f'x-->norm-->relu={out.size()}')

        out = self.convcap(out)
        if self.debug: print(f'convcap={out.size()}')

        x = self.shortcut(x)
        if self.debug: print(f'shortcut={x.size()}')

        d = self.num_cap
        out = torch.cat([x[:,:,:,:,:d,:]+out[:,:,:,:,:d,:], x[:,:,:,:,d:,:], out[:,:,:,:,d:,:]], 4)
        if self.debug: print(f'concatenation={out.size()}')

        return out


class DualPathBlock(nn.Sequential):
    def __init__(self, num_layers, input_num_cap, input_num_atom, num_cap, num_atom, growth_rate, stride):
        super(DualPathBlock, self).__init__()
        strides = [stride] + [1] * (num_layers - 1)
        last_num_cap = input_num_cap
        last_num_atom = input_num_atom

        for i, stride in enumerate(strides):
            first_layer = bool(i == 0)
            layer = _DualPathLayer(last_num_cap, last_num_atom, num_cap, num_atom, growth_rate, stride, first_layer)
            self.add_module(f'dualpathlayer{i+1}', layer)
            last_num_cap = num_cap + (i + 2) * growth_rate
            last_num_atom = num_atom
