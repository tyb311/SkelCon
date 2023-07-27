import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    @staticmethod
    def binary_focal(pr, gt, fov=None, gamma=2, *args):
        return -gt     *torch.log(pr)      *torch.pow(1-pr, gamma)
    def forward(self, pr, gt, fov=None, gamma=2, eps=1e-6, *args):
        pr = torch.clamp(pr, eps, 1-eps)
        loss1 = self.binary_focal(pr, gt)
        loss2 = self.binary_focal(1-pr, 1-gt)
        loss = loss1 + loss2
        return loss.mean()

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    @staticmethod
    def binary_cross_entropy(pr, gt, eps=1e-6):#alpha=0.25
        pr = torch.clamp(pr, eps, 1-eps)
        loss1 = -gt     *torch.log(pr) 
        loss2 = -(1-gt) *torch.log((1-pr))   
        return loss1, loss2 
        
    def forward(self, pr, gt, eps=1e-6, *args):#alpha=0.25
        loss1, loss2 = self.binary_cross_entropy(pr, gt) 
        return (loss1 + loss2).mean()#.item()

class DiceLoss(nn.Module):
    __name__ = 'DiceLoss'
    # DSC(A, B) = 2 * |A ^ B | / ( | A|+|B|)
    def __init__(self, ):
        super(DiceLoss, self).__init__()
        self.func = self.dice
    def forward(self, pr, gt, **args):
        return 2-self.dice(pr,gt)-self.dice(1-pr,1-gt)
        # return 1-self.func(pr, gt)
    @staticmethod
    def dice(pr, gt, smooth=1):#self, 
        pr,gt = pr.view(-1),gt.view(-1)
        inter = (pr*gt).sum()
        union = (pr+gt).sum()
        return (smooth + 2*inter) / (smooth + union)#*0.1


def get_loss(mode='fr'):
    print('loss:', mode)
    if mode=='fr':
        return FocalLoss()
    elif mode=='ce':
        return BCELoss()
    elif mode=='di':
        return DiceLoss()
    elif mode=='l2':
        return nn.MSELoss(reduction='mean')
        
    else:
        raise NotImplementedError()



