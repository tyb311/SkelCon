# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import sys
sys.path.append('.')
sys.path.append('..')

from utils import *
from nets import *
from scls import *


class SIAM(nn.Module):
	__name__ = 'siam'
	def __init__(self,
				 encoder,
				 clloss='nce',
				 temperature=0.1,
				 proj_num_layers=2,
				 pred_num_layers=2,
				 proj_num_length=64,
				 **kwargs):
		super().__init__()
		if clloss in CLLOSSES:
			self.loss = CLLOSSES[clloss]
		else:
			self.regular = self.sphere
			self.loss = get_sphere(clloss, proj_num_length)
		self.encoder = encoder
		# self.__name__ = self.encoder.__name__
		self.__name__ = 'X'.join([self.__name__, self.encoder.__name__]) #, clloss
		
		self.temperature = temperature
		self.proj_num_layers = proj_num_layers
		self.pred_num_layers = pred_num_layers

		self.projector = self.encoder.projector
		self.predictor = self.encoder.predictor

	def forward(self, img, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		out = self.encoder(img, **args)
		self.pred = self.encoder.pred
		self.feat = self.encoder.feat
		# self._dequeue_and_enqueue(proj1_ng, proj2_ng)
		if hasattr(self.encoder, 'tmp'):
			self.tmp = self.encoder.tmp
		return out

	def constraint(self, **args):
		return self.encoder.constraint(**args)

	def sphere(self, sampler, lab, fov=None):#contrastive loss split by classification
		feat = sampler.select(self.feat, self.pred.detach(), lab, fov)
		feat = self.projector(feat)
		self.proj = feat
		true = torch.zeros(size=(feat.shape[0],), dtype=torch.long).to(feat.device)
		true[:feat.shape[0]//2] = 1
		# print('regular:', feat.shape, true.shape)
		return self.loss(feat, true)

	def regular(self, sampler, lab, fov=None):#contrastive loss split by classification
		feat = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		proj = self.projector(feat)
		self.proj = proj
		pred = self.predictor(proj)
		# random mapping
		# pred, proj = sampler.norm(pred, proj)
		# rand = torch.randn(64, 64, device=feat.device)
		# pred = F.normalize(pred @ rand, dim=-1)
		# proj = F.normalize(proj @ rand, dim=-1)

		# compute loss
		losSG1 = self.loss(pred, proj.detach(), temperature=self.temperature)
		losSG2 = self.loss(proj, pred.detach(), temperature=self.temperature)
		return losSG1 + losSG2
 

		# 只计算对齐、相似性
		# losAU = align_uniform(pred, proj.detach()) 

		#	全局对齐
		# glb_sim = - (pred.detach() @ proj.permute(1,0)).mean() - (proj.detach() @ pred.permute(1,0)).mean()
		# glb_agn = align_loss(pred, torch.flip(proj, dims=[0,])) + align_loss(proj, torch.flip(pred, dims=[0,]))
		# glb_mse = F.mse_loss(pred, torch.flip(proj, dims=[0,]))
		# # print('global:', glb_agn.shape, glb_sim.shape)
		# return glb_sim + glb_agn + glb_mse# + losAU# + losKL#

		# #	困难样本与易分样本对齐
		# pro_hig, pro_low = torch.chunk(proj, chunks=2, dim=0)
		# pre_hig, pre_low = torch.chunk(pred, chunks=2, dim=0)
		# hig = torch.cat([pro_hig, pre_hig], dim=0)
		# low = torch.cat([pro_low, pre_low], dim=0)
		# loc_sim = - (hig.detach() @ low.permute(1,0)).mean() - (low.detach() @ hig.permute(1,0)).mean()
		# loc_agn = align_loss(hig, low) + align_loss(hig, torch.flip(low, dims=[0,]))
		# loc_mse = F.mse_loss(hig, low)
		# return loc_sim + loc_agn + loc_mse# + losAU# + losKL#

	# def regular(self, sampler, lab, fov=None):#contrastive loss split by batchsize
	# 	pred1, pred2 = torch.chunk(self.pred.detach(), chunks=2, dim=0)
	# 	feat1, feat2 = torch.chunk(self.feat, chunks=2, dim=0)
	# 	true1, true2 = torch.chunk(lab      , chunks=2, dim=0)
	# 	mask1, mask2 = torch.chunk(fov      , chunks=2, dim=0)
	# 	# print(feat1.shape, feat2.shape, self.pred.shape, lab.shape, fov.shape)

	# 	feat1 = sampler.select(feat1, pred1, true1, mask1)
	# 	proj1 = self.projector(feat1)
	# 	pred1 = self.predictor(proj1)

	# 	feat2 = sampler.select(feat2, pred2, true2, mask2)
	# 	proj2 = self.projector(feat2)
	# 	pred2 = self.predictor(proj2)

	# 	# compute loss
	# 	losSG1 = self.loss(pred1, proj2.detach(), temperature=self.temperature) + \
	# 			 self.loss(pred2, proj1.detach(), temperature=self.temperature)
	# 	# losSG2 = self.loss(proj, pred.detach(), temperature=self.temperature)

	# 	# pred = torch.flip(pred, dims=[0])
	# 	# proj = torch.flip(proj, dims=[0])
	# 	# losSG3 = self.loss(pred, proj.detach(), temperature=self.temperature)

	# 	# losAU = align_uniform(pred, proj.detach()) 
	# 	return losSG1# + losAU

	# def regular(self, sampler, lab, fov=None):#contrastive loss split by batchsize, 07-19 I find this wrong
	# 		proj1 = [fh, fl] is wrong
	# 		proj1 = [f, b] is right
	# 	self.encoder.regular(sampler, lab, fov, return_loss=False)
	# 	pred = self.predictor(self.encoder.emb)

	# 	proj1, proj2 = torch.chunk(self.encoder.emb, chunks=2, dim=0)
	# 	pred1, pred2 = torch.chunk(pred, chunks=2, dim=0)
	# 	# print(img1.shape, img2.shape)

	# 	# compute loss
	# 	losSG = self.loss(pred1, proj2.detach(), temperature=self.temperature) \
	# 		+ self.loss(pred2, proj1.detach(), temperature=self.temperature)
	# 	# losAU = align_uniform(pred1, proj2.detach()) + align_uniform(pred2, proj1.detach())
	# 	return losSG# + losAU
	# 	# return losCL + losSG


from build import *
import cv2
if __name__ == '__main__':
	pred = cv2.imread('figures/z_pred.png', cv2.IMREAD_GRAYSCALE)[:512, :512]
	mask = cv2.imread('figures/z_mask.png', cv2.IMREAD_GRAYSCALE)[:512, :512]
	true = cv2.imread('figures/z_true.png', cv2.IMREAD_GRAYSCALE)[:512, :512]
	h, w = pred.shape

	pred = torch.from_numpy(pred.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	mask = torch.from_numpy(mask.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	true = torch.from_numpy(true.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	print('imread:', pred.shape, mask.shape, true.shape)

	feat = torch.rand(1,32,h,w)
	feat = F.normalize(feat, p=2, dim=1)

	


	# net = SIAM(encoder=lunet(), clloss='sim2')
	net = SeqNet('lunet', 'lunet')
	net = SIAM(encoder=net, clloss='sim2', proj_num_length=32)
	# net.eval()
	ys = net(torch.rand_like(pred))

	for y in ys:
		print(y.shape)
	# print(net.__name__, y['loss'])

	# # net.train()

	# sampler = MLPSampler(top=4, low=0, mode='prob')
	# l = net.regular(sampler, pred, mask)
	# # l = net.regular3(sampler, pred, mask)
	# print(net.__name__, l.item())


	# l = net.sphere(sampler, pred, mask)
	# print(net.__name__, l.item())
	# plot4(net.feat)