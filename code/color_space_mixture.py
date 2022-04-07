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



def random_channel(rgb, tran=None):
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
	def forward(pic):  #random channel mixture
		a = random_channel(pic['img'])
		b = random_channel(pic['img'])
		alpha = random.random()
		pic['img'] = (alpha*a + (1-alpha)*b).astype(np.uint8)
		return pic