
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


