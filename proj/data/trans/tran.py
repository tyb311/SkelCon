
# tranDemo = transforms.Compose([
    # ToPILImage(),
    # ToTensor(),
    # ToNorm(mean=[0], std=[1]),
    # # Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    # ])
# Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
# Lambda(lambda crops: torch.stack([Normalize([0.485], [0.229])(crop) for crop in crops])),
        
# 常用资源库
import pandas as pd
import numpy as np
EPS = 1e-6#np.spacing(1)#
import os,glob,numbers

# 图像处理
import math,cv2,random, socket
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 图像显示
import matplotlib as mpl
if 'TAN' not in socket.gethostname():
    print('Run on Server!!!')
    mpl.use('Agg')#服务器绘图
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as f
