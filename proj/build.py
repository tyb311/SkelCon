
# 常用资源库
import pandas as pd
import numpy as np
EPS = 1e-9#
import os,glob,numbers
# 图像处理
import math,cv2,random
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from nets import lunet, SIAM


class MlpNorm(nn.Module):
	def __init__(self, dim_inp=256, dim_out=64):
		super(MlpNorm, self).__init__()
		dim_mid = min(dim_inp, dim_out)#max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_mid))
		linear_hidden.append(nn.Dropout(p=0.2))
		linear_hidden.append(nn.BatchNorm1d(dim_mid))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

		self.linear_out = nn.Linear(dim_mid, dim_out)# if num_layers >= 1 else nn.Identity()

	def forward(self, x):
		x = self.linear_hidden(x)
		x = self.linear_out(x)
		return F.normalize(x, p=2, dim=-1)

def torch_dilation(x, ksize=3, stride=1):
	return F.max_pool2d(x, (ksize, ksize), stride, ksize//2)

class MorphBlock(nn.Module):
	def __init__(self, inp_ch=2, channel=8):
		super().__init__()
		self.ch_wv = nn.Sequential(
			nn.Conv2d(inp_ch,channel,kernel_size=5, padding=2),
			nn.Conv2d(channel,channel,kernel_size=5, padding=2),
			nn.BatchNorm2d(channel),
			nn.Conv2d(channel,channel//2,kernel_size=3, padding=1),
		)
		self.ch_wq = nn.Sequential(
			nn.Conv2d(channel//2,8,kernel_size=3, padding=1),
			nn.BatchNorm2d(8),
			nn.Conv2d(8,1,kernel_size=1),
			nn.Sigmoid()
		)
	
	def forward(self, x, o):
		x = torch.cat([torch_dilation(o, ksize=3), x, o], dim=1)#, 1-torch_dilation(1-x, ksize=3)
		x = self.ch_wv(x)
		# print(x.shape)
		return self.ch_wq(x)

class SeqNet(nn.Module):#Supervised contrastive learning segmentation network
	__name__ = 'scls'
	def __init__(self, type_net, type_seg, num_emb=128):
		super(SeqNet, self).__init__()

		self.fcn = eval(type_net+'(num_emb=num_emb)')#build_model(cfg['net']['fcn'])
		self.seg = eval(type_seg+'(inp_c=32)')#build_model(cfg['net']['seg'])

		self.projector = MlpNorm(32, num_emb)#self.fcn.projector#MlpNorm(32, 64, num_emb)
		self.predictor = MlpNorm(num_emb, num_emb)#self.fcn.predictor#MlpNorm(32, 64, num_emb)

		self.morpholer1 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.morpholer2 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.__name__ = '{}X{}'.format(self.fcn.__name__, self.seg.__name__)

	def constraint(self, aux=None, fun=None, **args):
		aux = torch_dilation(aux)
		los1 = fun(self.sdm1, aux)
		los2 = fun(self.sdm2, aux)
		# if self.__name__.__contains__('dmf'):
		# 	los1 = los1 + self.fcn.regular()*0.1
		return los1, los2
	
	def regular(self, sampler, lab, fov=None, return_loss=True):
		emb = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projector(emb)
		# print(emb.shape)
		self.emb = emb
		if return_loss:
			return sampler.infonce(emb)
	tmp = {}
	def forward(self, x):
		aux = self.fcn(x)
		self.feat = self.fcn.feat
		out = self.seg(self.feat)
		self.pred = out
		# print(self.fcn.feat.shape, self.seg.feat.shape)
		self.sdm1 = self.morpholer1(self.fcn.feat, aux)
		self.sdm2 = self.morpholer2(self.seg.feat, out)
		self.tmp = {'sdm1':self.sdm1, 'sdm2':self.sdm2}

		if self.training:
			if isinstance(aux, (tuple, list)):
				return [self.pred, aux[0], aux[1]]
			else:
				return [self.pred, aux]
		return self.pred


def build_model(type_net='lunet', type_seg='lunet', type_loss='sim2', type_arch='', num_emb=128):
	# model = eval(type_net+'(num_emb=num_emb)')
	model = lunet(num_emb=num_emb)
		# raise NotImplementedError(f'--> Unknown type_net: {type_net}')

	if type_seg!='':
		model = SeqNet(type_net, type_seg, num_emb=num_emb)
		model = SIAM(encoder=model, clloss=type_loss, proj_num_length=num_emb)
	return model
