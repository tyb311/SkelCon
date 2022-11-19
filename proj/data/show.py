import torch
# from torchvision import models, transforms, utils, datasets
from torchvision.utils import make_grid as tvgrid

import numpy as np
np.set_printoptions(precision=4)

import math
import matplotlib.pyplot as plt
plt.rcParams['image.cmap']='gray'
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# 显示tensor shape
#---------------------------------------------------------------------------------------------
def desc(X, shape=False, value=True):
    def parse(X):
        if shape:
            print(X.shape, X.dtype)
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if value:
            print('X=[{}&{}]\tmean={}\tstd={}'.format(np.min(X), np.max(X), np.mean(X), np.std(X)))

    if isinstance(X, list):
        for i in X:
            parse(i)
    else:
        parse(X)

def to255(ret):
    return 255*ret
def gain(ret, p=1):    #gain_off
    mean = ret.mean()
    ret_min = mean-(mean-ret.min())*p
    ret_max = mean+(ret.max()-mean)*p
    ret = 255*(ret - ret_min)/(ret_max - ret_min)
    # ret = np.clip(ret, 0, 255).astype(np.uint8)
    # ret = ret.clip(0, 255)
    return ret

# 显示tensor图像
#---------------------------------------------------------------------------------------------
def show(X):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = gain(X.type(torch.float32))
    grid1 = tvgrid(X, nrow=1).numpy().astype(np.uint8)#.transpose((1, 2, 0))
    # print(grid1.shape, grid2.shape)
    if len(grid1.shape)==3 and grid1.shape[0] in [1,3]:
        grid1 = grid1.transpose((1, 2, 0))
    # plt.figure()
    plt.imshow(grid1, interpolation='nearest')#
    # plt.axis('off')
    plt.show()

def show2(X, Y, nrow=None):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
 
    X = gain(X)
    Y = gain(Y)
    # X = 255*X
    # Y = 255*Y

    if nrow is None:
        nrow = int(math.sqrt(X.shape[0])) if len(X.shape)>3 else 1
        # print('nrow:', nrow)
    grid1 = tvgrid(X, nrow=nrow).numpy().astype(np.uint8)#.transpose((1, 2, 0))
    grid2 = tvgrid(Y, nrow=nrow).numpy().astype(np.uint8)#.transpose((1, 2, 0))

    # print(grid1.shape, grid2.shape)
    if len(grid1.shape)==3 and grid1.shape[0] in [1,3]:
        grid1 = grid1.transpose((1, 2, 0))
        grid2 = grid2.transpose((1, 2, 0))
    # else:
    # print(grid1.shape, grid2.shape)

    # plt.figure()
    plt.subplot(121),plt.title('img'),plt.imshow(grid1, interpolation='nearest')#
    plt.subplot(122),plt.title('lab'),plt.imshow(grid2, interpolation='nearest')#
    # plt.axis('off')
    plt.show()


def show3(X, Y, Z, nrow=3, titles=['1','2','3'], raw=False):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        Z = torch.from_numpy(Z)
    if not raw:
        X = gain(X)
        Y = gain(Y)
        Z = gain(Z)
    # print('Tensor:', X.shape, Y.shape)
    grid1 = tvgrid(X, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid2 = tvgrid(Y, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid3 = tvgrid(Z, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))

    # plt.figure()
    plt.subplot(131),plt.title(titles[0]),plt.imshow(grid1, interpolation='nearest')#
    plt.subplot(132),plt.title(titles[1]),plt.imshow(grid2, interpolation='nearest')#
    plt.subplot(133),plt.title(titles[2]),plt.imshow(grid3, interpolation='nearest')#
    # plt.axis('off')
    plt.show()

def show4(X, Y, Z, W, nrow=2, titles=['1','2','3','4'], raw=False):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)
    if isinstance(Z, np.ndarray):
        Z = torch.from_numpy(Z)
    if isinstance(W, np.ndarray):
        W = torch.from_numpy(W)
    if raw:
        X = to255(X)
        Y = to255(Y)
        Z = to255(Z)
        W = to255(W)
    else:
        X = gain(X)
        Y = gain(Y)
        Z = gain(Z)
        W = gain(W)
    # print('Tensor:', X.shape, Y.shape)
    grid1 = tvgrid(X, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid2 = tvgrid(Y, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid3 = tvgrid(Z, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid4 = tvgrid(W, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))

    # plt.figure()
    plt.subplot(221),plt.title(titles[0]),plt.imshow(grid1, interpolation='nearest')#
    plt.subplot(222),plt.title(titles[1]),plt.imshow(grid2, interpolation='nearest')#
    plt.subplot(223),plt.title(titles[2]),plt.imshow(grid3, interpolation='nearest')#
    plt.subplot(224),plt.title(titles[3]),plt.imshow(grid4, interpolation='nearest')#
    # plt.axis('off')
    plt.show()

def show6(X, Y, Z, X1, Y1, Z1, nrow=3, titles=['1','2','3','1','2','3']):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        Z = torch.from_numpy(Z)

    X = gain(X)
    Y = gain(Y)
    Z = gain(Z)
    X1 = gain(X1)
    Y1 = gain(Y1)
    Z1 = gain(Z1)
    # print('Tensor:', X.shape, Y.shape)
    grid1 = tvgrid(X, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid2 = tvgrid(Y, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid3 = tvgrid(Z, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid11 = tvgrid(X1, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid21 = tvgrid(Y1, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))
    grid31 = tvgrid(Z1, nrow=nrow).numpy().astype(np.uint8).transpose((1, 2, 0))

    # plt.figure()
    plt.subplot(231),plt.title(titles[0]),plt.imshow(grid1, interpolation='nearest')#
    plt.subplot(232),plt.title(titles[1]),plt.imshow(grid2, interpolation='nearest')#
    plt.subplot(233),plt.title(titles[2]),plt.imshow(grid3, interpolation='nearest')#
    plt.subplot(234),plt.title(titles[3]),plt.imshow(grid11, interpolation='nearest')#
    plt.subplot(235),plt.title(titles[4]),plt.imshow(grid21, interpolation='nearest')#
    plt.subplot(236),plt.title(titles[5]),plt.imshow(grid31, interpolation='nearest')#
    # plt.axis('off')
    plt.show()