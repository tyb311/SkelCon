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



def similar_matrix2(q, k, temperature=0.1):#负太多了
	# print('similar_matrix2:', q.shape, k.shape)
	qfh, qfl, qbh, qbl = torch.chunk(q, 4, dim=0)
	kfh, kfl, kbh, kbl = torch.chunk(k, 4, dim=0)

	# negative logits: NxK
	l_pos = torch.einsum('nc,kc->nk', [qfl, kfh])
	l_neg = torch.einsum('nc,kc->nk', [qbl, kbh])
	# print(l_pos.shape, l_neg.shape)
	return 2 - l_pos.mean() - l_neg.mean()

CLLOSSES = {
	'sim2':similar_matrix2, 'sim3':similar_matrix2,
	}




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

	
	net = SIAM(encoder=lunet(), clloss='sim2')

	sampler = MLPSampler(top=4, low=0, mode='prob')


	# net.eval()
	ys = net(torch.rand_like(pred))

	for y in ys:
		print(y.shape)
	# print(net.__name__, y['loss'])

	# net.train()
	l = net.regular(sampler, pred, mask)
	# l = net.regular3(sampler, pred, mask)
	print(net.__name__, l.item())


	# plot(net.emb)