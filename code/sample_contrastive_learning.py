import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

top=4
low=1
dis=0
num=512


def points_selection(feat, prob, true, card=512, dis=100, **args):
    #point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	with torch.no_grad():
		prob = prob[true>.5].view(-1)
		idx = torch.sort(prob, dim=-1, descending=True)[1]

	h = torch.index_select(feat, dim=0, index=idx[:card])
	l = torch.index_select(feat, dim=0, index=idx[-card:])
	# print('lh:', l.shape, h.shape)

	return h, l



def feature_select(feat, pred, true, mask=None, ksize=5):
    # 'sample selection!'
    # print(feat.shape, true.shape)
    assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
    assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
    # reshape embeddings
    feat = feat.clone().permute(0,2,3,1).reshape(-1, feat.shape[1])

    true = true.round()
    dilate = F.max_pool2d(true, (ksize, ksize), stride=1, padding=ksize//2).round()
    edge = (dilate - true).round()
    back = (1-dilate).round()*mask.round()

    fh, fl = points_selection(feat,   pred, true, top=top, low=low, dis=dis, card=num*2)
    eh, el = points_selection(feat, 1-pred, edge, top=top, low=low, dis=dis, card=num)
    bh, bl = points_selection(feat, 1-pred, back, top=top, low=low, dis=dis, card=num)
    # print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)

    return torch.cat([fh, fl, eh, el, bh, bl], dim=0)



def similar_matrix2(q, k, temperature=0.1):
	# print('similar_matrix2:', q.shape, k.shape)
	qfh, qfl, qbh, qbl = torch.chunk(q, 4, dim=0)
	kfh, kfl, kbh, kbl = torch.chunk(k, 4, dim=0)

	# negative logits: NxK
	l_pos = torch.einsum('nc,kc->nk', [qfl, kfh])
	l_neg = torch.einsum('nc,kc->nk', [qbl, kbh])
	# print(l_pos.shape, l_neg.shape)
	return 2 - l_pos.mean() - l_neg.mean()




