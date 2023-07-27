
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import sys
sys.path.append('.')
sys.path.append('..')

from nets import *


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

		# compute loss
		losSG1 = self.loss(pred, proj.detach(), temperature=self.temperature)
		losSG2 = self.loss(proj, pred.detach(), temperature=self.temperature)
		return losSG1 + losSG2
 

