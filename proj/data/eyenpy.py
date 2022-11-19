# coding: utf-8
import numpy as np
EPS = 1e-9#np.spacing(1)
import os,glob,math,cv2,random,numbers
from torchvision import transforms
from torchvision.transforms import functional as f
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

# # Dataset
# http://matlabserver.cs.rug.nl/
# [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/) 
# [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) 
# [TONGREN](http://111.zbj99.cn/list.php?pid=3)

# 两种数据划分方式： 标准划分（如下）			随机对半划分（一半一半，随机挑选）
# drive   :			trainset:20 & testset:20
# chase   :			trainset:20 & testset:8
# stare   :			trainset:19 & testset:1
# hrf     :			trainset:15 & testset:30

#           图像大小  数量（对） 是否分组    训练集   验证集   测试集
# drive:    565x584     40         yes      15      5         20
# chase:    999x960     28         no       15      5         8
# stare:    700x605     20         no       7       3         10          


import socket
from albumentations import PadIfNeeded, CenterCrop, Resize

class EyeSetResource(object):
	size = dict()
	def __init__(self, folder=r'G:\Objects\expSeg\datasets\seteye', dbname='drive', isBasedPatch=False, **args):
		super(EyeSetResource, self).__init__()
		print('\n', '#'*32)

		if os.path.isdir('/home/tan/datasets/seteye'):
			self.folder = '/home/tan/datasets/seteye'
		elif os.path.isdir(r'G:\Objects\expSeg\datasets\seteye'):
			self.folder = r'G:\Objects\expSeg\datasets\seteye'
		else:
			# raise EnvironmentError('No thi root!')
			self.folder = folder

		self.dbname = dbname
		self.isBasedPatch = isBasedPatch
		self.imgs, self.labs, self.fovs, self.skes = self.getDataSet(self.dbname)#, self.mats
		self.lens = {'train':len(self.labs['train']),   'val':len(self.labs['val']),
					 'test':len(self.labs['test']),     'full':len(self.labs['full'])}  
		print('Number of Dataset:', self.lens)  
		if self.lens['test']>0:
			lab = self.readArr(self.labs['test'][0])
			self.size['raw'] = lab.shape
			h,w = lab.shape
			self.size['pad'] = (math.ceil(h/32)*32, math.ceil(w/32)*32)
			print('size:', self.size)
			hp,wp = self.size['pad']
			hc,wc = self.size['raw']
			self.tran_pade = PadIfNeeded(p=1, min_height=hp, min_width=wp)
			self.tran_crop = CenterCrop(height=hc, width=wc)
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
		if self.isBasedPatch:
			imgs_train = self.readFolder(dbname, part='patch', image='rgb')
			labs_train = self.readFolder(dbname, part='patch', image='lab')
			fovs_train = self.readFolder(dbname, part='patch', image='fov')
			skes_train = self.readFolder(dbname, part='patch', image='ske')
		else:
			imgs_train = self.readFolder(dbname, part='train', image='rgb')
			labs_train = self.readFolder(dbname, part='train', image='lab')
			fovs_train = self.readFolder(dbname, part='train', image='fov')
			skes_train = self.readFolder(dbname, part='train', image='ske')

		imgs_eval = self.readFolder(dbname, part='train', image='rgb')
		labs_eval = self.readFolder(dbname, part='train', image='lab')
		fovs_eval = self.readFolder(dbname, part='train', image='fov')
		skes_eval = self.readFolder(dbname, part='train', image='ske')

		# 全集
		imgs_full,labs_full,fovs_full,skes_full = [],[],[],[]
		imgs_full.extend(imgs_train); imgs_full.extend(imgs_test)
		labs_full.extend(labs_train); labs_full.extend(labs_test)
		fovs_full.extend(fovs_train); fovs_full.extend(fovs_test)
		skes_full.extend(skes_train); skes_full.extend(skes_test)

		db_imgs = {'train': imgs_train, 'val':imgs_eval, 'test': imgs_test, 'full':imgs_full}
		db_labs = {'train': labs_train, 'val':labs_eval, 'test': labs_test, 'full':labs_full}
		db_fovs = {'train': fovs_train, 'val':fovs_eval, 'test': fovs_test, 'full':fovs_full}
		db_skes = {'train': skes_train, 'val':skes_eval, 'test': skes_test, 'full':skes_full}
		return db_imgs, db_labs, db_fovs, db_skes

	def readFolder(self, dbname, part='train', image='rgb'):
		path = self.folder + '/' + dbname + '/' + part + '_' + image
		# if self.isBasedPatch:
		# 	imgs = glob.glob(path + '/*.png')
		# else:
		imgs = glob.glob(path + '/*.npy')
		imgs.sort()
		return imgs
		
	def readArr(self, image):
		# assert(image.endswith('.npy'), 'not npy file!') 
		# if self.isBasedPatch:
		# 	return np.array(Image.open(image))
		return np.load(image) 
	
	def readDict(self, index, exeData):  
		img = self.readArr(self.imgs[exeData][index])
		lab = self.readArr(self.labs[exeData][index])
		fov = self.readArr(self.fovs[exeData][index])
		ske = self.readArr(self.skes[exeData][index])
		# print(img.shape, lab.shape, fov.shape)
		if fov.shape[-1]==3:
			fov = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
		# mat = self.readArr(self.mats[exeData][index])
		return {'img':img, 'lab':lab, 'fov':fov, 'ske':ske}#
#end#


def imshow(*srs, nrow=1, titles=None, suptitle='image'):
	num = len(srs)
	plt.suptitle(suptitle)
	for i in range(num):
		plt.subplot(nrow,num//nrow,i+1)
		plt.imshow(srs[i])
		if isinstance(titles, list):
			plt.title(titles[i])
	plt.show()




def dataset2h5():
	# 想要把数据集存储为HDF5，却发现不方便并行读取
	with h5py.File('eye_'+db.dbname, 'w') as f:
		sub_db = f.create_group('train')
			
		for i in range(db.lens['train']):
			rgb = db.imread(db.imgs['train'][i], mode='RGB', array=True)
			lab = db.imread(db.labs['train'][i], mode='L', array=True)
			fov = db.imread(db.fovs['train'][i], mode='L', array=True)
			print(i, rgb.shape)
			gray = cv2.cvtColor(rgb, code=cv2.COLOR_RGB2GRAY)
			mat = MatchFilter.apply_filters(gray)
			imshow(rgb, mat, lab, fov, nrow=2)

			db_img = sub_db.create_group('id'+str(i))
			db_img.create_dataset('rgb', data=rgb)
			db_img.create_dataset('ske', data=mat)
			db_img.create_dataset('lab', data=lab)
			db_img.create_dataset('fov', data=fov)
				
		sub_db = f.create_group('test')
		for i in range(db.lens['test']):
			rgb = db.imread(db.imgs['test'][i], mode='RGB', array=True)
			lab = db.imread(db.labs['test'][i], mode='L', array=True)
			fov = db.imread(db.fovs['test'][i], mode='L', array=True)
			print(i, rgb.shape)
			gray = cv2.cvtColor(rgb, code=cv2.COLOR_RGB2GRAY)
			mat = MatchFilter.apply_filters(gray)
			# imshow(rgb, mat, lab, fov, nrow=2)

			db_img = sub_db.create_group('id'+str(i))
			db_img.create_dataset('rgb', data=rgb)
			db_img.create_dataset('ske', data=mat)
			db_img.create_dataset('lab', data=lab)
			db_img.create_dataset('fov', data=fov)


def dataset2npy(db):
	def to255(x):
		if x.max()<2:
			print('!!!!!!!!!!!!!!!!!!!!!!!!!')
		return x
		# maxval = 1 if x.max()>2 else 255
		# return x*maxval

	# 把数据集零散存为npz
	for i in range(db.lens['full']):
		pathimg = db.imgs['full'][i]
		pathlab = db.labs['full'][i]
		pathfov = db.fovs['full'][i]
		rgb = db.readArr(pathimg)
		lab = db.readArr(pathlab)
		fov = db.readArr(pathfov)

		if db.dbname=='hrf':
			rgb = cv2.resize(rgb, dsize=(fov.shape[0]//2, fov.shape[1]//2))
			lab = cv2.resize(lab, dsize=(fov.shape[0]//2, fov.shape[1]//2))
			fov = cv2.resize(fov, dsize=(fov.shape[0]//2, fov.shape[1]//2))

		# gray = cv2.cvtColor(rgb, code=cv2.COLOR_RGB2GRAY)
		# mat = MatchFilter.apply_filters(gray)
		# imshow(rgb, mat, lab, fov)
		# assert(rgb.shape==3, 'rgb shape!=3')
		# assert(lab.shape==2, 'lab shape!=2')
		# assert(fov.shape==2, 'rgb shape!=2')
		# print(rgb.shape, lab.shape, fov.shape)

		# pics = db.readDict(i, 'train')
		# a,b,c = pics['img'],pics['lab'],pics['fov']#,d,pics['ske']
		print(pathimg)
		pathimg = pathimg[:-4].replace('images', 'rgb')
		pathlab = pathlab[:-4].replace('manual', 'lab')#.replace('.', '_')
		pathfov = pathfov[:-4].replace('marsker', 'fov')
		# print(pathimg,pathlab,pathfov)

		for path in [pathimg, pathlab, pathfov]:
			folder = db.dbname + '/' + path.split('\\')[0]
			if not os.path.exists(folder):
				print(folder)
				os.makedirs(folder)

		cv2.imwrite(pathimg.replace('..', '.')+'.png', rgb)
		cv2.imwrite(pathlab.replace('..', '.')+'.png', lab)
		cv2.imwrite(pathfov.replace('..', '.')+'.png', fov)
		# np.save(pathimg.replace('..', '.'), to255(rgb))
		# np.save(pathlab.replace('..', '.'), to255(lab))
		# np.save(pathfov.replace('..', '.'), to255(fov))

# 主函数
#---------------------------------------------------------------------------------------------
'''滑动窗口'''
def grid_crop(image, stride=64, ksize=128, prefix='lab', count=0):
	height, width = image.shape[:2]
	height = int(np.ceil(height/32.0)*32)
	width = int(np.ceil(width/32.0)*32)

	cnt = 0
	for y in range(0, height, stride):
		for x in range(0, width, stride):
			if (y+ksize)<=height and (x+ksize)<=width:#没超出下边界，也超出下边界
				isBasedPatch = image[y:y+ksize, x:x+ksize]
				path_save = '{}/patch_{}/{}{}_{}.png'.format(r"G:\Objects\expSeg\datasets\seteye\drive", prefix, prefix, count, cnt)
				# cv2.imwrite(path_save, isBasedPatch)
				path_save = '{}/patch_{}/{}{}_{}'.format(r"G:\Objects\expSeg\datasets\seteye\drive", prefix, prefix, count, cnt)
				np.save(path_save, isBasedPatch)
				cnt += 1
				print(cnt, isBasedPatch.shape, path_save)
			if (y+ksize)>height and (x+ksize)>width:#超出右边界，但没超出下边界 或者 超出下边界，但没超出右边界
				continue
			if (y+ksize)>height and (x+ksize)<=width:#超出下边界，也超出下边界
				break 
			
def crop4trainset(stride=64, ksize=128):
	# db = EyeSetResource(folder='../eyeset', dbname='drive', isBasedPatch=True)
	db = EyeSetResource(folder='../eyeset', dbname='drive', isBasedPatch=False)
	# db = EyeSetResource(folder='../eyeset', dbname='stare')

	for i in range(db.lens['train']):
		pics = db.readDict(i, 'train')
		a,b,c = pics['img'],pics['lab'],pics['fov']#,d,pics['ske']
		# print(i, a.shape)
		# imshow(a,b,c, nrow=1)#,d
		# grid_crop(a, prefix='rgb', count=i)
		# grid_crop(b, prefix='lab', count=i)
		# grid_crop(c, prefix='fov', count=i)
	

		h_raw, w_raw = a.shape[:2]
		height, width = db.size['pad']
		num_pixel = height * width
		a = db.tran_pade(image=a)['image']
		b = db.tran_pade(image=b)['image']
		c = db.tran_pade(image=c)['image']
		cnt = 0
		for y in range(0, height, stride):
			for x in range(0, width, stride):
				if (y+ksize)<=height and (x+ksize)<=width:#没超出下边界，也超出下边界
					patcha = a[y:y+ksize, x:x+ksize]
					patchb = b[y:y+ksize, x:x+ksize]
					patchc = c[y:y+ksize, x:x+ksize]
					if patchb.sum()>1 and patchc.sum()*2>num_pixel:
						path_save = '{}/patch_{}/{}{}_{}'.format(r"G:\Objects\expSeg\datasets\seteye\drive", 'lab', 'lab', i, cnt)
						np.save(path_save, patchb)

						path_save = '{}/patch_{}/{}{}_{}'.format(r"G:\Objects\expSeg\datasets\seteye\drive", 'rgb', 'rgb', i, cnt)
						np.save(path_save, patcha)
						
						path_save = '{}/patch_{}/{}{}_{}'.format(r"G:\Objects\expSeg\datasets\seteye\drive", 'fov', 'fov', i, cnt)
						np.save(path_save, patchc)

						cnt += 1
						print(cnt, patchb.shape, path_save)
				if (y+ksize)>height and (x+ksize)>width:#超出右边界，但没超出下边界 或者 超出下边界，但没超出右边界
					continue
				if (y+ksize)>height and (x+ksize)<=width:#超出下边界，也超出下边界
					break 


from skimage import morphology
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
def main():
	# print(os.path.isdir(r'G:\Objects\LabGpu'))
	# from tran_filter import *
	import h5py
	# 默认增强后数据集为白色血管
	# db = EyeSetResource(folder='eyeset', dbname='hrf', npy=True)
	# EyeSetResource(folder='eyeset', dbname='chase', npy=False)
	# EyeSetResource(folder='eyeset', dbname='hrf', npy=False)
	# EyeSetResource(folder='eyeset', dbname='stare', npy=False)
	# db = EyeSetResource(folder='../eyeset', dbname='chase')
	db = EyeSetResource(folder='../eyeset', dbname='hrf')
	# db = EyeSetResource(folder='G:\Objects\datasets\seteye', dbname='stare')
	# db = EyeSetResource(folder='../eyeset', dbname='stare')

	# dataset2npy(db)
	mode = 'val'
	mode = 'test'
	tag = 'train' if mode=='val' else 'test'
	for i in range(db.lens[mode]):
		pics = db.readDict(i, mode)
		a,b,c = pics['img'],pics['lab'],pics['fov']#,d,pics['ske']
		print(i, a.shape, b.min(), b.max(), c.min(), c.max())

		skel = morphology.skeletonize((pics['lab']/255.0).round()).astype(np.uint8)
		d = morphology.dilation(skel, kernel)*255
		cv2.imwrite(r'G:\Objects\datasets\seteye\stare\{}_ske\{:02d}_skel.png'.format(tag, i), d)
		np.save(r'G:\Objects\datasets\seteye\stare\{}_ske\{:02d}_skel'.format(tag,i), d)

		# cv2.imwrite(r'G:\Objects\datasets\seteye\drive\test_ske\{:02d}_skel.png'.format(i), d)
		# np.save(r'G:\Objects\datasets\seteye\stare\test_ske\{:02d}_skel'.format(i), d)
		# imshow(a,b,c,d, nrow=2)#
	# mode = 'train'
	# print(mode)
	# for i,p in enumerate(dbname.imgs['train']):
	#     img = Image.open(p)
	#     img = np.array(img)
	#     h5_path = p.split('.')[0]+'.h5'
	#     h5_path = h5_path.replace('eyeset', 'eyeset_h5')
	#     print(i, img.shape, h5_path)
	#     f = h5py.File(h5_path, 'w')
	#     d = f.create_dataset(name=mode, shape=img.shape, dtype=np.uint8)#compression=gzip
	#     d[i:i+1,...] = np.zeros(shape=img.shape, dtype=np.uint8)
	#     f.close()

	# print('val')
	# for p in dbname.imgs['val']:
	#     print(p)
	# print('test')
	# for p in dbname.imgs['test']:
	#     print(p)
	# for p in dbname.imgs['full']:
	#     print(p)



if __name__ == '__main__':
	# main()
	# crop4trainset()		

	db = EyeSetResource(folder='G:\Objects\datasets\seteye', dbname='hrf')
	# # db = EyeSetResource(folder='../eyeset', dbname='stare')

	dataset2npy(db)
	# mode = 'val'
	# # mode = 'test'
	# folder = r'G:/Objects/datasets/seteye/hrf3x2'
	# for i in range(db.lens[mode]):
	# 	pics = db.readDict(i, mode)
	# 	a,b,c,d = pics['img'],pics['lab'],pics['fov'],pics['ske']
	# 	a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
	# 	print(i, a.shape, b.shape, c.shape, d.shape)

	# 	skel = morphology.skeletonize((pics['lab']/255.0).round()).astype(np.uint8)
	# 	d = morphology.dilation(skel, kernel)*255

	# 	tag = 'train' if mode=='val' else 'test'
	# 	cv2.imwrite(folder + '/{}_ske/{:02d}.png'.format(tag, i), d)
	# 	np.save(folder + '/{}_ske/{:02d}'.format(tag, i), d)
