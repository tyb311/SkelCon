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

from data import *
from nets import *
from build import *

from grad import *
from loop import *

#start#
import argparse
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="Train network")
#	实验参数	C:对比学习、P：先验知识
parser.add_argument('--inc', type=str, default='', help='instruction')#skeleton & dilation
parser.add_argument('--gpu', type=int, default=0, help='cuda number')
parser.add_argument('--los', type=str, default='fr', help='loss function')
parser.add_argument('--net', type=str, default='lunet', help='network')
parser.add_argument('--seg', type=str, default='lunet', help='network')
# parser.add_argument('--patch', type=str2bool, default=True, help='Patch based!')
parser.add_argument('--csm', type=str2bool, default=True, help='Color Space Mixture!')
parser.add_argument('--coff_ds', type=float, default=0.5, help='Cofficient of Deep Supervision!')

#	数据参数
parser.add_argument('--db', type=str, default='drive', help='instruction')
parser.add_argument('--loo', type=int, default=0, help='Leave One Out')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--ds', type=int, default=128, help='data size')
parser.add_argument('--pl', type=str2bool, default=False, help='Parallel!')
parser.add_argument('--root', type=str, default='', help='root folder')
#	正则化参数
parser.add_argument('--ct', type=str2bool, default=True, help='Constraint for Network!')
parser.add_argument('--coff_ct', type=float, default=.9, help='Cofficient of Constraint!')
parser.add_argument('--loss_ct', type=str, default='di', help='Loss of Contrastive learning!')

#	对比学习相关参数
parser.add_argument('--arch', type=str, default='siam', help='architechture')
parser.add_argument('--roma', type=str2bool, default=False, help='Random Mapping!')
# parser.add_argument('--rd', type=str2bool, default=False, help='Render for Contrastive Learning!')
parser.add_argument('--coff_cl', type=float, default=.1, help='Cofficient of Contrastive learning!')
parser.add_argument('--temp_cl', type=float, default=.1, help='Temperature of Contrastive learning!')
parser.add_argument('--loss_cl', type=str, default='sim3', help='Loss of Contrastive learning!')#, choices=['', 'au', 'nce', 'sim', 'nce2', 'sim2']

parser.add_argument('--sss', type=str, default='half', choices=['', 'hard', 'half'], help='Sample Selection Strategy!')
parser.add_argument('--top', type=int, default=4, help='sampler top')
parser.add_argument('--low', type=int, default=2, help='sampler low')
parser.add_argument('--dis', type=int, default=4, help='sampler dis')
parser.add_argument('--num', type=int, default=512, help='sampler number')
# parser.add_argument('--se3', type=str2bool, default=False, help='Select 3 or 2!')
# parser.add_argument('--con', type=str, default='', help='infonce version')

args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(args.gpu, 4)])

# 训练程序########################################################
if __name__ == '__main__':
	args.inc += ('_csm' if args.csm else '')

	dataset = EyeSetGenerator(dbname=args.db, datasize=args.ds, loo=args.loo) 
	# dataset = EyeSetGenerator(dbname=args.db, isBasedPatch=args.patch) 
	dataset.use_csm = args.csm

	net = build_model(args.net, args.seg, args.loss_cl, args.arch)
	if args.db=='stare':
		net.__name__ += 'LOO'+str(args.loo)

	keras = KerasTorch(model=net, args=args) 
	keras.args = args
	keras.isParallel = args.pl

	args.ct = False

	net.__name__ += args.inc + 'ds'+str(args.coff_ds) + args.sss
	if args.sss!='':
		args.se3 = args.loss_cl.endswith('3')
		keras.sampler = MLPSampler(mode=args.sss, select3=args.se3, roma=args.roma,
			top=args.top, low=args.low, dis=args.dis, num=args.num)
		if args.sss=='semi':
			net.__name__ += str(args.dis)
		elif args.sss in ['part', 'prob']:
			net.__name__ += str(args.top) + str(args.low)	#Top & low
		elif args.sss in ['thsh', 'drop']:#, 'hard'
			net.__name__ += str(args.dis)
			
	print('Network Name:', net.__name__)

	keras.compile(dataset, loss=args.los, lr=0.01)  
	keras.gradUtil.coff_ds = args.coff_ds
	if args.root=='':
		keras.test(testset=dataset, key='los', flagSave=True)  
		keras.val()
		keras.fit(epochs=169)    

	keras.test(testset=dataset, key='los', flagSave=True, tta=False)  
	for key in keras.paths.keys():
		keras.test(testset=dataset, key=key, flagSave=True, tta=False)

