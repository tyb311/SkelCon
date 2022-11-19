import torch
import torch.nn as nn
import torch.nn.functional as F


def points_selection_drop(feat, prob, true, card=512, dis=0, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	assert dis>=0 and dis<5, 'discard number must be in range(0,5)'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.sort(prob, dim=-1, descending=False)[1]
	hard_ranks = list(torch.chunk(idx, chunks=5, dim=0))
	
	hard_ranks.pop(dis)
	idx = torch.cat(hard_ranks, dim=0)

	rand = torch.randperm(idx.shape[0])
	sample1 = idx[rand[:card]]
	# print(prob[sample][:9])
	sample2 = idx[rand[-card:]]
	# print(prob[sample][:9])
	############################################################

	h = torch.index_select(feat, dim=0, index=sample1)
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l
	
def points_selection_semi(feat, prob, true, card=512, dis=0, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	assert dis>=0 and dis<=2, 'discard number of semi can only be 0,1 or 2!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.sort(prob, dim=-1, descending=False)[1]
	idx10 = torch.chunk(idx, chunks=10, dim=0)
	idx_h = torch.cat(idx10[4+dis:8+dis], dim=0)
	idx_l = torch.cat(idx10[dis:4+dis], dim=0)
	# print(idx_h.shape, idx_l.shape)
	sample1 = idx_h[torch.randperm(idx_h.shape[0])[:card]]
	sample2 = idx_l[torch.randperm(idx_l.shape[0])[:card]]
	############################################################
	# print(prob[sample][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	# print(prob[sample][:9])
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_thsh(feat, prob, true, card=512, dis=6, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	dis = dis*0.1
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.arange(0, prob.numel(), device=feat.device)

	idx_h = idx[prob> dis]
	idx_l = idx[prob<=dis]

	sample1 = idx_h[torch.randperm(idx_h.shape[0])[:card]]
	sample2 = idx_l[torch.randperm(idx_l.shape[0])[:card]]
	############################################################
	# print(prob[sample1][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	# print(prob[sample2][:9])
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_prob(feat, prob, true, card=512, top=4, low=2, **args):#按照概率划分为4份
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	assert top>=0 and top<=4, 'top number can only be 0-4!'
	assert low>=0 and low<=4 and low<top, 'top number can only be 0-4!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	top = top * 0.2
	low = low * 0.2
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.arange(0, prob.numel(), device=feat.device)

	idx_h = idx[(prob>=top) & (prob<top+0.2)]
	idx_l = idx[(prob>=low) & (prob<low+0.2)]

	sample1 = idx_h[torch.randperm(idx_h.shape[0])[:card]]
	sample2 = idx_l[torch.randperm(idx_l.shape[0])[:card]]
	############################################################
	# print(top, top+0.2, low, low+0.2)
	# print(prob[sample1][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	# print(prob[sample2][:9])
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_rand(feat, prob, true, card=512, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.sort(prob, dim=-1, descending=False)[1]
	rand = torch.randperm(idx.shape[0])
	sample1 = idx[rand[:card]]
	# print(prob[sample][:9])
	sample2 = idx[rand[-card:]]
	# print(prob[sample][:9])
	############################################################
	h = torch.index_select(feat, dim=0, index=sample1)
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_part(feat, prob, true, card=512, top=4, low=2, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	assert top>=0 and top<5, 'top must be in range(0,5)'
	assert low>=0 and low<5, 'low must be in range(0,5)'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.sort(prob, dim=-1, descending=False)[1]
	hard_ranks = torch.chunk(idx, chunks=5, dim=0)
	sample1 = hard_ranks[top][torch.randperm(hard_ranks[top].shape[0])[:card]]
	# print(prob[sample][:9])
	sample2 = hard_ranks[low][torch.randperm(hard_ranks[low].shape[0])[:card]]
	# print(prob[sample][:9])
	############################################################
	h = torch.index_select(feat, dim=0, index=sample1)
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

def points_selection_hard(feat, prob, true, card=512, dis=100, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.sort(prob, dim=-1, descending=True)[1]
	# h = torch.index_select(feat, dim=0, index=idx[dis:dis+card])
	# l = torch.index_select(feat, dim=0, index=idx[-dis-card:-dis])
	############################################################
	h = torch.index_select(feat, dim=0, index=idx[:card])
	l = torch.index_select(feat, dim=0, index=idx[-card:])
	# print('lh:', l.shape, h.shape)
	# print(prob[idx[:card]].view(-1)[:9])
	# print(prob[idx[-card:]].view(-1)[:9])
	return h, l

def points_selection_half(feat, prob, true, card=512, **args):#point selection by ranking
	assert len(feat.shape)==2, 'feat should contains N*L two dims!'
	L = feat.shape[-1]
	# print(feat.shape, true.shape)
	feat = feat[true.view(-1,1).repeat(1, L)>.5].view(-1, L)
	
	############################################################
	# with torch.no_grad():
	prob = prob[true>.5].view(-1)
	idx = torch.sort(prob, dim=-1, descending=False)[1]
	idx_l, idx_h = torch.chunk(idx, chunks=2, dim=0)

	sample1 = idx_h[torch.randperm(idx_h.shape[0])[:card]]
	sample2 = idx_l[torch.randperm(idx_l.shape[0])[:card]]
	############################################################
	# print(prob[sample][:9])
	h = torch.index_select(feat, dim=0, index=sample1)
	# print(prob[sample][:9])
	l = torch.index_select(feat, dim=0, index=sample2)
	# print('lh:', l.shape, h.shape)
	return h, l

class MLPSampler:
	func = points_selection_half
	def __init__(self, mode='hard', top=4, low=1, dis=0, num=512, select3=False, roma=False):
		self.top = top
		self.low = low
		self.dis = dis
		self.num = num
		self.roma = roma
		self.select = self.select3 if select3 else self.select2

		self.func = eval('points_selection_'+mode)

	@staticmethod
	def rand(*args):
		return MLPSampler(mode='rand', num=512).select(*args)
	@staticmethod
	def half(*args):
		return MLPSampler(mode='half', num=512).select(*args)
	def norm(self, *args, roma=False):
		# if self.roma or roma:# random mapping
		# 	dim = args[0].shape[-1]
		# 	rand = torch.randn(dim, dim, device=args[0].device)
		# 	args = [F.normalize(arg @ rand, dim=-1) for arg in args]
		args = [F.normalize(arg, dim=-1) for arg in args]
		if len(args)==1:
			return args[0]
		return args

	def select(self, feat, pred, true, mask=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		# print(feat.shape, true.shape)
		assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.clone().permute(0,2,3,1).reshape(-1, feat.shape[1])
		true = true.round()
		fh, fl = self.func(feat,   pred, true, top=self.top, low=self.low, dis=self.dis, card=self.num)
		return torch.cat([fh, fl], dim=0)

	def select2(self, feat, pred, true, mask=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.permute(0,2,3,1).reshape(-1, feat.shape[1])
		# feat = F.normalize(feat, p=2, dim=-1)
		true = true.round()
		back = (F.max_pool2d(true, (ksize, ksize), 1, ksize//2) - true).round()
		# back = (1-true).round()*mask.round()

		fh, fl = self.func(feat,   pred, true, top=self.top, low=self.low, dis=self.dis, card=self.num)
		bh, bl = self.func(feat, 1-pred, back, top=self.top, low=self.low, dis=self.dis, card=self.num)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		# return [fh, fl, bh, bl]
		# print(mode)
		return torch.cat([fh, fl, bh, bl], dim=0)

	def select3(self, feat, pred, true, mask=None, ksize=5):
		# assert mode in ['hard', 'semi', 'hazy', 'edge'], 'sample selection mode wrong!'
		# print(feat.shape, true.shape)
		assert feat.shape[-2:]==true.shape[-2:], 'shape of feat & true donot match!'
		assert feat.shape[-2:]==pred.shape[-2:], 'shape of feat & pred donot match!'
		# reshape embeddings
		feat = feat.clone().permute(0,2,3,1).reshape(-1, feat.shape[1])
		# feat = F.interpolate(feat, size=true.shape[-2:], mode='bilinear', align_corners=True)
		# feat = F.normalize(feat, p=2, dim=-1)
		true = true.round()
		dilate = F.max_pool2d(true, (ksize, ksize), stride=1, padding=ksize//2).round()
		edge = (dilate - true).round()
		back = (1-dilate).round()*mask.round()

		# plt.subplot(131),plt.imshow(true.squeeze().data.numpy())
		# plt.subplot(132),plt.imshow(edge.squeeze().data.numpy())
		# plt.subplot(133),plt.imshow(back.squeeze().data.numpy())
		# plt.show()

		# assert back.sum()>0, 'back has no pixels!'
		# assert true.sum()>0, 'true has no pixels!'
		# print('true:', true.sum().item(), true.sum().item()/true.numel())
		# print('edge:', edge.sum().item(), edge.sum().item()/edge.numel())
		# print('back:', back.sum().item(), back.sum().item()/back.numel())
		# print(feat.shape, pred.shape, true.shape)

		fh, fl = self.func(feat,   pred, true, top=self.top, low=self.low, dis=self.dis, card=self.num*2)
		eh, el = self.func(feat, 1-pred, edge, top=self.top, low=self.low, dis=self.dis, card=self.num)
		bh, bl = self.func(feat, 1-pred, back, top=self.top, low=self.low, dis=self.dis, card=self.num)
		# print('mlp_sample:', fh.shape, fl.shape, bh.shape, bl.shape)
		# return [fh, fl, bh, bl]
		# print(mode)
		return torch.cat([fh, fl, eh, el, bh, bl], dim=0)
		# return torch.cat([fh, fl, eh, bh, el, bl], dim=0)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def plot4(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, init='pca', random_state=2021)
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )
	[fh, fl, bh, bl] = np.split(emb, 4, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(fh[:, 0], fh[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(fl[:, 0], fl[:, 1], c='pink', marker='^', linewidths=.1)
	plt.scatter(bh[:, 0], bh[:, 1], c='green', marker='o', linewidths=.1)
	plt.scatter(bl[:, 0], bl[:, 1], c='lightgreen', marker='o', linewidths=.1, alpha=0.7)
	plt.title('embedded feature distribution')
	plt.legend(['vessel_easy', 'vessel_hard', 'retinal_easy', 'retinal_hard'])
	# plt.legend(['vessel_h', 'vessel_l', 'retinal_h', 'retinal_l'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig

def plot2(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, random_state=2021, init='pca')
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )

	[f, b] = np.split(emb, 2, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(f[:, 0], f[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(b[:, 0], b[:, 1], c='green', marker='o', linewidths=.1)
	plt.title('embedded feature distribution')
	plt.legend(['vessel', 'retinal'])
	# plt.legend(['vessel_h', 'vessel_l', 'retinal_h', 'retinal_l'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig


def plot6(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, init='pca', random_state=2021)
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )
	[fh, fl, eh, el, bh, bl] = np.split(emb, 6, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(fh[:, 0], fh[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(fl[:, 0], fl[:, 1], c='pink', marker='^', linewidths=.1)
	plt.scatter(eh[:, 0], eh[:, 1], c='purple', marker='o', linewidths=.1)
	plt.scatter(el[:, 0], el[:, 1], c='violet', marker='o', linewidths=.1)
	plt.scatter(bh[:, 0], bh[:, 1], c='green', marker='o', linewidths=.1)
	plt.scatter(bl[:, 0], bl[:, 1], c='lightgreen', marker='o', linewidths=.1, alpha=0.7)
	plt.title('embedded feature distribution')
	plt.legend(['vessel_easy', 'vessel_hard', 'edge_easy', 'edge_hard', 'retinal_easy', 'retinal_hard'])
	# plt.legend(['vessel_h', 'vessel_l', 'retinal_h', 'retinal_l'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig

def plot3(emb=None, path_save='emb.png'):
	# assert len(emb[0].shape)==2, 'no embedding!'
	# for f in emb:
	# 	print('EMB:', f.shape)
	print('save embedding!')
	tsne = TSNE(n_components=2, init='pca', random_state=2021)
	emb = tsne.fit_transform(emb.detach().cpu().data.numpy() )
	[f, e, b] = np.split(emb, 3, axis=0)

	# print(fh.shape, fl.shape, bh.shape, bl.shape)
	fig = plt.figure()
	plt.scatter(f[:, 0], f[:, 1], c='red', marker='^', linewidths=.1)
	plt.scatter(e[:, 0], e[:, 1], c='purple', marker='o', linewidths=.1)
	plt.scatter(b[:, 0], b[:, 1], c='green', marker='o', linewidths=.1)
	plt.title('embedded feature distribution')
	plt.legend(['vessel', 'edge', 'retinal'])
	plt.tight_layout()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig('./draw_examples3.pdf', format='pdf')
	plt.savefig(path_save, dpi=90)
	# plt.show()
	return fig



'''
或许构建正负样本要少考虑些假阳性样本，简单的做法是：
	不考虑负样本，只在正样本上做对比学习，且不考虑背景类，我开始不相信专家了
'''
import cv2
if __name__ == '__main__':
	pred = cv2.imread('figures/z_pred.png', cv2.IMREAD_GRAYSCALE)
	true = cv2.imread('figures/z_true.png', cv2.IMREAD_GRAYSCALE)
	mask = cv2.imread('figures/z_mask.png', cv2.IMREAD_GRAYSCALE)
	h, w = pred.shape

	pred = torch.from_numpy(pred.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	true = torch.from_numpy(true.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	mask = torch.from_numpy(mask.astype(np.float32)/255).unsqueeze(0).unsqueeze(0)
	# print('imread:', pred.shape, mask.shape, true.shape)

	feat = torch.rand(1,32,h,w)
	feat = F.normalize(feat, p=2, dim=1)

	smp = MLPSampler(top=1, low=4, mode='part')
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)

	# args = smp.norm(emb)
	# print(args[0].shape)

	smp = MLPSampler(top=2, low=4, mode='part')
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)

	smp = MLPSampler(mode='drop', dis=1)
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)

	smp = MLPSampler(mode='prob', top=4, low=0)
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)

	smp = MLPSampler(mode='thsh', top=4, low=0)
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)

	smp = MLPSampler(mode='half', top=4, low=0)
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)

	smp = MLPSampler(mode='hard')
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)
	smp = MLPSampler(mode='hard', select3=True)
	emb = smp.select(feat, pred, true, mask)
	print(emb.shape)

	# smp = MLPSampler(mode='semi', dis=0)
	# emb = smp.select(feat, pred, true, mask)
	# print(emb.shape)

	# smp = MLPSampler(mode='semi', dis=1)
	# emb = smp.select(feat, pred, true, mask)
	# print(emb.shape)

	# smp = MLPSampler(mode='semi', dis=2)
	# emb = smp.select(feat, pred, true, mask)
	# print(emb.shape)

	# emb = MLPSampler.rand(feat, pred, true, mask)
	# print(emb.shape)

	# emb = MLPSampler.half(feat, pred, true, mask)
	# print(emb.shape)


	# plot6(emb)
	# plt.show()
	# plot3(emb)
	# plt.show()
	# import time
	# st = time.time()
	# plot2(emb)
	# print('time:', time.time() - st)
	# plt.show()
	# umap-time: 10.627314805984497
	# tsne-time: 5.323570966720581


	# a = torch.rand(64, 3)
	# b = list(torch.chunk(a, 8, dim=0))
	# b.pop(2)
	# c = torch.cat(b, dim=0)
	# print(a.shape, c.shape)