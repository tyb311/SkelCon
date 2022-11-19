# -*- encoding:utf-8 -*-
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
import socket
import matplotlib as mpl
if 'TAN' not in socket.gethostname():
	print('Run on Server!!!')
	mpl.use('Agg')#服务器绘图

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def gain(ret, p=1):    #gain_off
	mean = np.mean(ret)
	ret_min = mean-(mean-np.min(ret))*p
	ret_max = mean+(np.max(ret)-mean)*p
	ret = 255*(ret - ret_min)/(ret_max - ret_min)
	ret = np.clip(ret, 0, 255).astype(np.uint8)
	return ret

def arr2img(pic):
	return Image.fromarray(pic.astype(np.uint8))#, mode='L'

def arrs2imgs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = arr2img(pic[key])
	return _pic

def imgs2arrs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = np.array(pic[key])
	return _pic

def pil_tran(pic, tran=None):
	if tran is None:
		return pic
	if isinstance(tran, list):
		for t in tran:
			for key in pic.keys():
				pic[key] = pic[key].transpose(t)
	else:
		for key in pic.keys():
			pic[key] = pic[key].transpose(tran)
	return pic

# class Aug4Val(object):
#     number = 8
#     @staticmethod
#     def forward(pic, flag):
#         flag %= Aug4Val.number
#         if flag==0:
#             return pic
#         pic = arrs2imgs(pic)
#         if flag==1:
#             return imgs2arrs(pil_tran(pic, tran=Image.FLIP_LEFT_RIGHT))
#         if flag==2:
#             return imgs2arrs(pil_tran(pic, tran=Image.FLIP_TOP_BOTTOM))
#         if flag==3:
#             return imgs2arrs(pil_tran(pic, tran=Image.ROTATE_180))
#         if flag==4:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE]))
#         if flag==5:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.FLIP_TOP_BOTTOM]))
#         if flag==6:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.FLIP_LEFT_RIGHT]))
#         if flag==7:
#             return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.ROTATE_180]))
class Aug4Val(object):
	number = 4
	@staticmethod
	def forward(pic, flag):
		flag %= Aug4Val.number
		if flag==0:
			return pic
		pic = arrs2imgs(pic)
		if flag==1:
			return imgs2arrs(pil_tran(pic, tran=Image.FLIP_LEFT_RIGHT))
		if flag==2:
			return imgs2arrs(pil_tran(pic, tran=Image.FLIP_TOP_BOTTOM))
		if flag==3:
			return imgs2arrs(pil_tran(pic, tran=Image.ROTATE_180))
		# if flag==4:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))
		# if flag==5:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_TOP_BOTTOM]))
		# if flag==6:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT]))
		# if flag==7:
		# 	return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))

def random_channel(rgb, tran=None):##cv2.COLOR_RGB2HSV,HSV不好#
	if tran is None:
		tran = random.choice([
			cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
			cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
			cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
			cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
			])
	# if rgb.shape[-1]!=3:#单通道图片不做变换
	#     return rgb
	rgb = cv2.cvtColor(rgb, tran)
	if tran==cv2.COLOR_RGB2LAB:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2XYZ:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2LUV:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2HLS:
		rgb = cv2.split(rgb)[1]
	elif tran==cv2.COLOR_RGB2YCrCb:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2YUV:
		rgb = cv2.split(rgb)[0]
	elif tran==cv2.COLOR_RGB2BGR:
		rgb = cv2.split(rgb)[1]
		# rgb = random.choice(cv2.split(rgb))#
	return rgb

class Aug4CSA(object):#Color Space Augment
	number = 8
	trans = [
			cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
			cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
			cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
			cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
			]
	@staticmethod
	def forward(pic, flag):
		flag %= Aug4CSA.number
		pic['img'] = random_channel(pic['img'], tran=Aug4CSA.trans[flag])
		return pic
	@staticmethod
	def forward_train(pic):  #random channel mixture
		a = random_channel(pic['img'])
		b = random_channel(pic['img'])
		alpha = random.random()
		pic['img'] = (alpha*a + (1-alpha)*b).astype(np.uint8)
		return pic

class EyeSetResource(object):
	size = dict()
	def __init__(self, folder='../eyeset', dbname='drive', loo=None, **args):
		super(EyeSetResource, self).__init__()
		
		if os.path.isdir('/home/tan/datasets/seteye'):
			self.folder = '/home/tan/datasets/seteye'
		elif os.path.isdir('/home/tyb/datasets/seteye'):
			self.folder = '/home/tyb/datasets/seteye'
		else:
			self.folder = '../datasets/seteye'
		# else:
		# 	raise EnvironmentError('No thi root!')
		# self.folder = folder
		self.dbname = dbname

		self.imgs, self.labs, self.fovs, self.skes = self.getDataSet(self.dbname)
		if dbname=='stare' and loo is not None: 
			self.imgs['test'] = [self.imgs['full'][loo]]
			self.imgs['train'] = self.imgs['full'][:loo] + self.imgs['full'][1+loo:]
			self.imgs['val'] = self.imgs['train']
			
			self.labs['test'] = [self.labs['full'][loo]]
			self.labs['train'] = self.labs['full'][:loo] + self.labs['full'][1+loo:]
			self.labs['val'] = self.labs['train']
			
			self.fovs['test'] = [self.fovs['full'][loo]]
			self.fovs['train'] = self.fovs['full'][:loo] + self.fovs['full'][1+loo:]
			self.fovs['val'] = self.fovs['train']
			
			self.skes['test'] = [self.skes['full'][loo]]
			self.skes['train'] = self.skes['full'][:loo] + self.skes['full'][1+loo:]
			self.skes['val'] = self.skes['train']
			print('LOO:', loo, self.imgs['test'])
			print('LOO:', loo, self.labs['test'])
			print('LOO:', loo, self.fovs['test'])
			print('LOO:', loo, self.skes['test'])

		self.lens = {'train':len(self.labs['train']),   'val':len(self.labs['val']),
					 'test':len(self.labs['test']),     'full':len(self.labs['full'])}  
		# print(self.lens)  
		if self.lens['test']>0:
			lab = self.readArr(self.labs['test'][0])
			self.size['raw'] = lab.shape
			h,w = lab.shape
			self.size['pad'] = (math.ceil(h/32)*32, math.ceil(w/32)*32)
			print('size:', self.size)
		else:
			print('dataset has no images!')

		# print('*'*32,'eyeset','*'*32)
		strNum = 'images:{}+{}+{}#{}'.format(self.lens['train'], self.lens['val'], self.lens['test'], self.lens['full'])
		print('{}@{}'.format(self.dbname, strNum))

	def getDataSet(self, dbname):        
		# 测试集
		imgs_test = self.readFolder(dbname, part='test', image='rgb')
		labs_test = self.readFolder(dbname, part='test', image='lab')
		fovs_test = self.readFolder(dbname, part='test', image='fov')
		skes_test = self.readFolder(dbname, part='test', image='ske')
		# 训练集
		imgs_train = self.readFolder(dbname, part='train', image='rgb')
		labs_train = self.readFolder(dbname, part='train', image='lab')
		fovs_train = self.readFolder(dbname, part='train', image='fov')
		skes_train = self.readFolder(dbname, part='train', image='ske')
		# 全集
		imgs_full,labs_full,fovs_full,skes_full = [],[],[],[]
		imgs_full.extend(imgs_train); imgs_full.extend(imgs_test)
		labs_full.extend(labs_train); labs_full.extend(labs_test)
		fovs_full.extend(fovs_train); fovs_full.extend(fovs_test)
		skes_full.extend(skes_train); skes_full.extend(skes_test)

		db_imgs = {'train': imgs_train, 'val':imgs_train, 'test': imgs_test, 'full':imgs_full}
		db_labs = {'train': labs_train, 'val':labs_train, 'test': labs_test, 'full':labs_full}
		db_fovs = {'train': fovs_train, 'val':fovs_train, 'test': fovs_test, 'full':fovs_full}
		db_skes = {'train': skes_train, 'val':skes_train, 'test': skes_test, 'full':skes_full}
		return db_imgs, db_labs, db_fovs, db_skes

	def readFolder(self, dbname, part='train', image='rgb'):
		path = self.folder + '/' + dbname + '/' + part + '_' + image
		imgs = glob.glob(path + '/*.npy')
		imgs.sort()
		return imgs
		
	def readArr(self, image):
		# assert(image.endswith('.npy'), 'not npy file!') 
		return np.load(image) 
	
	def readDict(self, index, exeData):  
		img = self.readArr(self.imgs[exeData][index])
		fov = self.readArr(self.fovs[exeData][index])
		lab = self.readArr(self.labs[exeData][index])
		ske = self.readArr(self.skes[exeData][index])
		if fov.shape[-1]==3:
			fov = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
		return {'img':img, 'lab':lab, 'fov':fov, 'ske':ske}#

import imgaug as ia
import imgaug.augmenters as iaa
IAA_NOISE = iaa.OneOf(children=[# Noise
		iaa.Add((-7, 7), per_channel=True),
		iaa.AddElementwise((-7, 7)),
		iaa.Multiply((0.9, 1.1), per_channel=True),
		iaa.MultiplyElementwise((0.9, 1.1), per_channel=True),

		iaa.AdditiveGaussianNoise(scale=3, per_channel=True),
		iaa.AdditiveLaplaceNoise(scale=3, per_channel=True),
		iaa.AdditivePoissonNoise(lam=5, per_channel=True),

		iaa.SaltAndPepper(0.01, per_channel=True),
		iaa.ImpulseNoise(0.01),
	]
)
IAA_BLEND = iaa.OneOf(children=[# Noise
		# Blend
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(seed=random.randint(0,9)), background=iaa.Multiply(seed=random.randint(0,9))),
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Add(seed=random.randint(0,9)), background=iaa.Add(seed=random.randint(0,9))),
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Multiply(seed=random.randint(0,9)), background=iaa.Multiply(seed=random.randint(0,9))),
		iaa.BlendAlpha((0.0, 1.0), foreground=iaa.Multiply(seed=random.randint(0,9)), background=iaa.Add(seed=random.randint(0,9))),

		iaa.BlendAlphaElementwise(.3, iaa.Clouds(), seed=random.randint(0,9), per_channel=True),
		iaa.BlendAlphaElementwise(.5, iaa.AddToBrightness(21), seed=random.randint(0,9), per_channel=True),
		iaa.BlendAlphaElementwise(.5, iaa.AddToHue(21), seed=random.randint(0,9), per_channel=True),
		iaa.BlendAlphaElementwise(.5, iaa.AddToSaturation(21), seed=random.randint(0,9), per_channel=True),

		iaa.BlendAlphaVerticalLinearGradient(iaa.Clouds(), max_value=.5, seed=random.randint(0,9)),
		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToSaturation((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaVerticalLinearGradient(iaa.AddToBrightness((-30, 30)), seed=random.randint(0,9)),

		iaa.BlendAlphaHorizontalLinearGradient(iaa.Clouds(), max_value=.5, seed=random.randint(0,9)),
		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToSaturation((-30, 30)), seed=random.randint(0,9)),
		iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToBrightness((-30, 30)), seed=random.randint(0,9)),

		iaa.BlendAlphaCheckerboard(7, 7, iaa.AddToHue((-30, 30)), seed=random.randint(0,9)),
	]
)
TRANS_NOISE = iaa.Sequential(children=[IAA_NOISE, IAA_BLEND])

from albumentations import (
	# 翻转
	Flip, Transpose, RandomRotate90, PadIfNeeded, RandomGridShuffle,
	# 其他
	OneOf, Compose, CropNonEmptyMaskIfExists, CLAHE, RandomGamma
) # 图像变换函数

TRANS_TEST = Compose([CLAHE(p=1), RandomGamma(p=1)])#
TRANS_AAUG = Compose([      
	OneOf([Transpose(p=1), RandomRotate90(p=1), ], p=.7),
	Flip(p=.7), 
])


from skimage import morphology
from torch.utils.data import DataLoader, Dataset
class EyeSetGenerator(Dataset, EyeSetResource):
	exeNums = {'train':Aug4CSA.number, 'val':Aug4Val.number, 'test':1}
	exeMode = 'train'#train, val, test
	exeData = 'train'#train, test, full

	SIZE_IMAGE = 128
	expCross = False   
	LEN_AUG = 32
	def __init__(self, datasize=128, **args):
		super(EyeSetGenerator, self).__init__(**args)
		self.SIZE_IMAGE = datasize
		self.LEN_AUG = 96 // (datasize//64)**2
		print('SIZE_IMAGE:{} & AUG SIZE:{}'.format(self.SIZE_IMAGE, self.LEN_AUG))
		
	def __len__(self):
		length = self.lens[self.exeData]*self.exeNums[self.exeMode]
		if self.isTrainMode:
			return length*self.LEN_AUG
		return length

	def set_mode(self, mode='train'):
		self.exeMode = mode
		self.exeData = 'full' if self.expCross else mode 
		self.isTrainMode = (mode=='train')
		self.isValMode = (mode=='val')
		self.isTestMode = (mode=='test')
	def trainSet(self, bs=8, data='train'):#pin_memory=True, , shuffle=True
		self.set_mode(mode='train')
		return DataLoader(self, batch_size=bs, pin_memory=True, num_workers=4, shuffle=True)
	def valSet(self, data='val'):
		self.set_mode(mode='val')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	def testSet(self, data='test'):
		self.set_mode(mode='test')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	#DataLoader worker (pid(s) 5220) exited unexpectedly, 令numworkers>1就可以啦
	
	# @staticmethod
	def parse(self, pics, cat=True):
		rows, cols = pics['lab'].squeeze().shape[-2:]     
		for key in pics.keys(): 
			# print(key, pics[key].shape)
			pics[key] = pics[key].view(-1,1,rows,cols) 
		return pics['img'], torch.round(pics['lab']), torch.round(pics['fov']), torch.round(pics['ske'])

	def post(self, img, lab, fov):
		if type(img) is not np.ndarray:img = img.squeeze().cpu().numpy()
		if type(lab) is not np.ndarray:lab = lab.squeeze().cpu().numpy()
		if type(fov) is not np.ndarray:fov = fov.squeeze().cpu().numpy()
		img = img * fov
		return img, lab, fov

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	# use_csm = True
	def __getitem__(self, idx, divide=32):
		index = idx % self.lens[self.exeData] 
		pics = self.readDict(index, self.exeData)
		imag = pics['img']# = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)

		# pics['aux'] = pics['ske']
		if self.isTrainMode:
			# print(pics['lab'].shape, pics['fov'].shape, pics['aux'].shape)
			mask = np.stack([pics['lab'], pics['fov'], pics['ske']], axis=-1)
			# 裁剪增强
			augCrop = CropNonEmptyMaskIfExists(p=1, height=self.SIZE_IMAGE, width=self.SIZE_IMAGE)
			picaug = augCrop(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']

			# 随机增强
			# if random.choice([True, False]):
			imag = TRANS_TEST(image=imag)['image']
			# if random.choice([True, False]):
			imag = TRANS_TEST(image=imag)['image']

			# 添加噪声
			imag = TRANS_NOISE(image=imag)
			# 变换增强
			picaug = TRANS_AAUG(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']
			
			pics['img'] = imag
			pics['lab'], pics['fov'], pics['ske'] = mask[:,:,0],mask[:,:,1],mask[:,:,2]
			if self.use_csm:
				pics = Aug4CSA.forward_train(pics)
		else:
			# pics['img'] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
			pics['img'] = TRANS_TEST(image=pics['img'])['image']
			# 图像补齐
			h, w = pics['lab'].shape
			w = int(np.ceil(w / divide)) * divide
			h = int(np.ceil(h / divide)) * divide
			augPad = PadIfNeeded(p=1, min_height=h, min_width=w)
			for key in pics.keys():
				pics[key] = augPad(image=pics[key])['image']

			if self.isValMode:# 验证增强->非测试，则增强
				flag = idx//self.lens[self.exeData]
				# pics = Aug4CSA.forward(pics, flag)
				pics = Aug4Val.forward(pics, flag)
			# elif self.isTestMode:	
				# pics = Aug4CSA.forward_test(pics)

		if pics['img'].shape[-1]==3:#	green or gray
			pics['img'] = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)
			# pics['img'] = pics['img'][:,:,1]#莫非灰度图像比绿色通道更好一点？

		# 骨架化
		# skel = morphology.skeletonize((pics['lab']/255.0).round()).astype(np.uint8)
		# pics['ske'] = morphology.dilation(skel, self.kernel)*255
		# pics['ske'] = pics['ske'] | pics['lab']	#与不与，这是个问题

		for key in pics.keys():
			# print(key, pics[key].shape)
			pics[key] = torch.from_numpy(pics[key]).type(torch.float32).div(255)
		return pics

def tensor2image(x):
	x = x.squeeze().data.numpy()
	return x

def torch_dilation(x, ksize=5, stride=1):
	return F.max_pool2d(x, (ksize, ksize), stride, ksize//2)
	# return F.max_pool2d(x, kernel_size=(3,3), stride=2)



if __name__ == '__main__':
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='drive', isBasedPatch=True)#
	db = EyeSetGenerator(folder='../datasets/seteye', dbname='hrf', isBasedPatch=True)#
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='chase', isBasedPatch=True)#
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='stare', loo=0)#
	# db = EyeSetGenerator(folder=r'G:\Objects\expSeg\datasets\seteye', dbname='drive', isBasedPatch=False)#
	# db.expCross = True
	print('generator:', len(db.trainSet()), len(db.valSet()), len(db.testSet()), )

	# db.expCross=True
	# for i, imgs in enumerate(db.trainSet(1)):
	for i, imgs in enumerate(db.valSet(1)):
	# for i, imgs in enumerate(db.testSet()):
		# print(imgs.keys())
		# print(imgs)
		(img, lab, fov, aux) = db.parse(imgs)
		print(img.shape, lab.shape, fov.shape, aux.shape)
		# fov = torch_dilation(lab)

		# img = torch_dilation(lab, ksize=3)
		# fov = 1-torch_dilation(1-lab, ksize=3)

		plt.subplot(221),plt.imshow(tensor2image(img))
		plt.subplot(222),plt.imshow(tensor2image(lab))
		plt.subplot(223),plt.imshow(tensor2image(fov))
		plt.subplot(224),plt.imshow(tensor2image(aux))
		plt.show()
