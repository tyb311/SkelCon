import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append('.')
sys.path.append('..')

from utils import *
from nets.modules import *

class SCNetConv2d(nn.Module):
	# 作者团队：南开大学(程明明组)b&NUS&字节AI Lab
	# http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
	# 代码链接：https://github.com/backseason/SCNet
	# SCNet：通过自校准卷积改进卷积网络
	def __init__(self, in_channels, out_channels, pooling_ratio=2, stride=1, **kwargs):
		super(SCNetConv2d, self).__init__()
		self.k2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=pooling_ratio, stride=pooling_ratio),
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(in_channels)
		)
		self.k3 = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(in_channels)
		)
		self.k4 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
		)
	def forward(self, x):
		out = torch.sigmoid(x + F.interpolate(self.k2(x), x.size()[2:]))
		out = torch.mul(self.k3(x), out)
		return self.k4(out)

class SpAttConv2d(nn.Module):
	"""Split-Attention nn.Conv2d . Idea from ResNeSt"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
				dilation=1, groups=1, radix=2, reduction_factor=4, **kwargs):
		super(SpAttConv2d, self).__init__()
		inter_channels = max(in_channels*radix//reduction_factor, 32)
		self.radix = radix
		self.cardinality = groups
		self.out_channels = out_channels
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels*radix, kernel_size, stride, padding, dilation,
							groups=groups*radix, bias=False),
			nn.BatchNorm2d(out_channels*radix),
			DisOut(),
			nn.ReLU(),
		)
		self.fc1 = nn.Sequential(
			nn.Conv2d(out_channels, inter_channels, 1, groups=self.cardinality),
			nn.BatchNorm2d(inter_channels),#由于BatchNorm操作需要多于一个数据计算平均值
			nn.ReLU(),
		)
		self.fc2 = nn.Conv2d(inter_channels, out_channels*radix, 1, groups=self.cardinality)

	def forward(self, x):
		x = self.conv(x)

		batch, channel = x.shape[:2]
		if self.radix > 1:
			splited = torch.split(x, channel//self.radix, dim=1)
			gap = sum(splited) 
		else:
			gap = x
		gap = F.adaptive_avg_pool2d(gap, 1)
		gap = self.fc1(gap)

		atten = self.fc2(gap).view((batch, self.radix, self.out_channels))
		if self.radix > 1:
			atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
		else:
			atten = torch.sigmoid(atten).view(batch, -1, 1, 1)

		if self.radix > 1:
			atten = torch.split(atten, channel//self.radix, dim=1)
			out = sum([att*split for (att, split) in zip(atten, splited)])
		else:
			out = atten * x
		return out.contiguous()

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(BasicConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#BasicConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()

		self.o = nn.Identity()
		drop_prob=0.15
		# self.o = DisOut(drop_prob=drop_prob)#
		self.o = nn.Dropout2d(p=drop_prob, inplace=False) 

		if activation=='frelu':
			self.a = FReLU(out_channels)
		elif activation is None:
			self.a = nn.Identity()
		else:
			self.a = activation

	def forward(self, x):
		x = self.c(x)# x = torch.clamp_max(x, max=99)
		# print('c:', x.max().item())
		x = self.o(x)
		# print('o:', x.max().item())
		x = self.b(x)
		# print('b:', x.max().item())
		x = self.a(x)
		# print('a:', x.max().item())
		return x

class DemoConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(DemoConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#BasicConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()
		self.a = nn.ReLU()

	def forward(self, x):
		x = self.c(x)# x = torch.clamp_max(x, max=99)
		# print('c:', x.max().item())
		x = self.b(x)
		# print('b:', x.max().item())
		x = self.a(x)
		# print('a:', x.max().item())
		return x

class PyridConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(PyridConv2d, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=PyConv3, groups=in_channels)
	def forward(self, x):
		return self.c(x)

class CDiff(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(CDiff, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=CDCConv, groups=in_channels)
	def forward(self, x):
		return self.c(x)

class DoverConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=F.gelu,
				):#ACTIVATION,nn.PReLU(),torch.sin
		super(DoverConv2d, self).__init__()
		self.c = BasicConv2d(in_channels=in_channels, out_channels=out_channels, 
				stride=stride, bn=True, conv=DoConv)
	def forward(self, x):
		return self.c(x)

class Bottleneck(nn.Module):
	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, downsample=None, **args):
		super(Bottleneck, self).__init__()

		self.conv1 = self.MyConv(in_c, out_c, kernel_size=3, stride=stride)
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
			# DisOut(),#prob=0.2
			nn.BatchNorm2d(out_c)
		)
		self.relu = swish#nn.LeakyReLU()
		if downsample is None and in_c != out_c:
			downsample = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
				# DisOut(),#prob=0.2
				nn.BatchNorm2d(out_c),
			)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		if self.downsample is not None:
			residual = self.downsample(x)
		# print('Basic:', x.shape)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.relu(out + residual)
		# print(out.min().item(), out.max().item())
		return out

class ConvBlock(torch.nn.Module):
	attention=None
	MyConv = BasicConv2d
	def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
		super(ConvBlock, self).__init__()
		self.shortcut = nn.Sequential(nn.Conv2d(inp_c, out_c, kernel_size=1), nn.BatchNorm2d(out_c))
		pad = (ksize - 1) // 2

		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		block = []
		block.append(self.MyConv(inp_c, out_c, kernel_size=ksize, padding=pad))

		if self.attention=='ppolar':
			# print('ppolar')
			block.append(ParallelPolarizedSelfAttention(out_c))
		elif self.attention=='spolar':
			# print('spolar')
			block.append(SequentialPolarizedSelfAttention(out_c))
		elif self.attention=='siamam':
			# print('siamam')
			block.append(simam_module(out_c))
		# else:
		# 	print(self.attention)
		block.append(self.MyConv(out_c, out_c, kernel_size=ksize, padding=pad))
		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return swish(out + self.shortcut(x))

# 输出层 & 下采样
class OutSigmoid(nn.Module):
	def __init__(self, inp_planes, out_planes=1, out_c=8):
		super(OutSigmoid, self).__init__()
		self.cls = nn.Sequential(
			nn.Conv2d(in_channels=inp_planes, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			# nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_c),
			nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1, bias=True),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.cls(x)


if __name__ == '__main__':
    channels = 2
    
    inputs = torch.randn(1,channels,128,256)
    
    
    net = CDiff(channels, 5)
    depth =  net(inputs)
    print(depth.shape)

    net = DoverConv2d(channels, 5)
    depth =  net(inputs)
    print(depth.shape)
    
    
    
    