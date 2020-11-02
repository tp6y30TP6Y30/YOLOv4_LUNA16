import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers.attention_layers import SEModule, CBAM

class Mish(nn.Module):
	def __init__(self):
		super(Mish, self).__init__()

	def forward(self, x):
		return x * torch.tanh(F.softplus(x))

norm_name = {"bn": nn.BatchNorm3d}
activate_name = {
	"relu": nn.ReLU,
	"leaky": nn.LeakyReLU,
	'linear': nn.Identity(),
	"mish": Mish()
}

class Conv_block(nn.Module):
	def __init__(self, filters_in, filters_out, kernel_size, stride = 1, norm = 'bn', activate = 'mish'):
		super(Conv_block, self).__init__()
		self.norm = norm
		self.activate = activate
		self.__conv = nn.Conv3d(in_channels = filters_in, out_channels = filters_out, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2, bias = not norm)

		if norm:
			if norm == "bn":
				self.__norm = norm_name[norm](num_features = filters_out)

		if activate:
			if activate == "leaky":
				self.__activate = activate_name[activate](negative_slope, inplace = True)
			if activate == "relu":
				self.__activate = activate_name[activate](inplace = True)
			if activate == "mish":
				self.__activate = activate_name[activate]

	def forward(self, x):
		x = self.__conv(x)
		if self.norm:
			x = self.__norm(x)
		if self.activate:
			x = self.__activate(x)
		return x

class CSPBlock(nn.Module):
	def __init__(self, in_channels, out_channels, hidden_channels = None, residual_activation = 'linear'):
		super(CSPBlock, self).__init__()

		if hidden_channels is None:
			hidden_channels = out_channels

		self.block = nn.Sequential(
			Conv_block(in_channels, hidden_channels, 1),
			Conv_block(hidden_channels, out_channels, 3)
		)

		self.activation = activate_name[residual_activation]
		self.attention = "None"
		if self.attention == 'SEnet':self.attention_module = SEModule(out_channels)
		elif self.attention == 'CBAM':self.attention_module = CBAM(out_channels)
		else: self.attention = None

	def forward(self, x):
		residual = x
		out = self.block(x)
		if self.attention is not None:
			out = self.attention_module(out)
		out += residual
		return out

class CSPFirstStage(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(CSPFirstStage, self).__init__()
		self.downsample_conv = Conv_block(in_channels, out_channels, 3, stride = 2)
		self.split_conv0 = Conv_block(out_channels, out_channels, 1)
		self.split_conv1 = Conv_block(out_channels, out_channels, 1)

		self.blocks_conv = nn.Sequential(
			CSPBlock(out_channels, out_channels, in_channels),
			Conv_block(out_channels, out_channels, 1)
		)

		self.concat_conv = Conv_block(out_channels * 2, out_channels, 1)

	def forward(self, x):
		x = self.downsample_conv(x)
		x0 = self.split_conv0(x)
		x1 = self.split_conv1(x)
		x1 = self.blocks_conv(x1)
		x = torch.cat([x0, x1], dim = 1)
		x = self.concat_conv(x)
		return x

class CSPStage(nn.Module):
	def __init__(self, in_channels, out_channels, num_blocks):
		super(CSPStage, self).__init__()
		self.downsample_conv = Conv_block(in_channels, out_channels, 3, stride = 2)
		self.split_conv0 = Conv_block(out_channels, out_channels // 2, 1)
		self.split_conv1 = Conv_block(out_channels, out_channels // 2, 1)

		self.blocks_conv = nn.Sequential(
			*[CSPBlock(out_channels // 2, out_channels // 2) for _ in range(num_blocks)],
			Conv_block(out_channels // 2, out_channels // 2, 1)
		)

		self.concat_conv = Conv_block(out_channels, out_channels, 1)

	def forward(self, x):
		x = self.downsample_conv(x)
		x0 = self.split_conv0(x)
		x1 = self.split_conv1(x)
		x1 = self.blocks_conv(x1)
		x = torch.cat([x0, x1], dim = 1)
		x = self.concat_conv(x)

		return x

class CSPDarknet53(nn.Module):
	def __init__(self, stem_channels = 32, feature_channels = [64, 64, 64, 64], num_features = 3): # feature_channels = [64, 128, 256, 512, 1024]
		super(CSPDarknet53, self).__init__()
		self.stem_conv = Conv_block(1, stem_channels, 1)
		self.stages = nn.ModuleList([
			CSPFirstStage(stem_channels, feature_channels[0]),
			CSPStage(feature_channels[0], feature_channels[1], 2),
			CSPStage(feature_channels[1], feature_channels[2], 8),
			CSPStage(feature_channels[2], feature_channels[3], 8),
			# CSPStage(feature_channels[3], feature_channels[4], 4)
		])

		self.feature_channels = feature_channels
		self.num_features = num_features
		self.__initialize_weight()

	def forward(self, x):
		x = self.stem_conv(x)
		features = []
		for stage in self.stages:
			x = stage(x)
			features.append(x)
		return features[-self.num_features:]

	def __initialize_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

def _BuildCSPDarknet53():
	model = CSPDarknet53()
	return model, model.feature_channels[-3:]

