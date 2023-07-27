import os, glob, sys, time, torch
from torch.optim import lr_scheduler
from nets import *
from optim import *
torch.set_printoptions(precision=3)

class GradUtil(object):
	def __init__(self, model, loss='ce', lr=0.01, wd=2e-4, root='.'):
		self.path_checkpoint = os.path.join(root, 'super_params.tar')
		if not os.path.exists(root):
			os.makedirs(root)

		self.lossName = loss
		self.criterion = get_loss(loss)
		params = filter(lambda p:p.requires_grad, model.parameters())
		self.optimizer = RAdamW(params=params, lr=lr, weight_decay=2e-4)
		self.scheduler = ReduceLR(name=loss, optimizer=self.optimizer,  
			mode='min', factor=0.7, patience=2, 
			verbose=True, threshold=0.0001, threshold_mode='rel', 
			cooldown=2, min_lr=1e-5, eps=1e-9)
		
	def isLrLowest(self, thresh=1e-5):
		return self.optimizer.param_groups[0]['lr']<thresh

	coff_ds = 0.5
	def calcGradient(self, criterion, outs, true, fov=None):
		lossSum = 0#torch.autograd.Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
		if isinstance(outs, (list, tuple)):
			# ratio = 1/(1+len(outs))
			for i in range(len(outs)-1,0,-1):#第一个元素尺寸最大
				# print('输出形状：', outs[i].shape, true.shape)
				# true = torch.nn.functional.interpolate(true, size=outs[i].shape[-2:], mode='nearest')
				loss = criterion(outs[i], true)#, fov
				lossSum = lossSum + loss*self.coff_ds
			outs = outs[0]
		# print(outs.shape, true.shape)
		lossSum = lossSum + criterion(outs, true)#, fov
		return lossSum
		
	def backward_seg(self, pred, true, fov=None, model=None, requires_grad=True, losInit=[]):
		self.optimizer.zero_grad()

		costList = []
		#torch.autograd.Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
		los = self.calcGradient(self.criterion, pred, true, fov)
		costList.append(los)
		self.total_loss += los.item()
		del pred, true, los

		if isinstance(losInit, list) and len(losInit)>0:#hasattr(losInit, 'item'):#not isinstance(losInit, int):
			costList.extend(losInit)

		losSum = sum(costList)
		losStr = ','.join(['{:.4f}'.format(los.item()) for los in costList])
		if requires_grad:
			losSum.backward()#梯度归一化
			#梯度裁剪
			# nn.utils.clip_grad_value_(model.parameters(), clip_value=1.1)#clip_value=1.1
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)#（最大范数，L2)
			self.optimizer.step()
		return losSum.item(), losStr

	total_loss = 0
	def update_scheduler(self, i=0):
		logStr = '\r{:03}# '.format(i)
		# losSum = 0
		logStr += '{}={:.4f},'.format(self.lossName, self.total_loss)
		print(logStr, end='')
		# self.callBackEarlyStopping(los=losSum)

		if isinstance(self.scheduler, ReduceLR):
			self.scheduler.step(self.total_loss)
		else:
			self.scheduler.step()
		self.total_loss = 0