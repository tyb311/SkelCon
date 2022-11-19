import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

class FReLU(nn.Module):
	r""" FReLU formulation, with a window size of kxk. (k=3 by default)"""
	def __init__(self, in_channels, *args):
		super().__init__()
		self.func = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
			nn.BatchNorm2d(in_channels)
		)
	def forward(self, x):
		x = torch.max(x, self.func(x))
		return x

class DisOut(nn.Module):
	def __init__(self, drop_prob=0.5, block_size=6, alpha=1.0):
		super(DisOut, self).__init__()

		self.drop_prob = drop_prob      
		self.weight_behind=None
  
		self.alpha=alpha
		self.block_size = block_size
		
	def forward(self, x):
		if not self.training: 
			return x

		x=x.clone()
		if x.dim()==4:           
			width=x.size(2)
			height=x.size(3)

			seed_drop_rate = self.drop_prob* (width*height) / self.block_size**2 / (( width -self.block_size + 1)*( height -self.block_size + 1))
			
			valid_block_center=torch.zeros(width,height,device=x.device).float()
			valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),int(self.block_size // 2):(height - (self.block_size - 1) // 2)]=1.0
			valid_block_center=valid_block_center.unsqueeze(0).unsqueeze(0)
			
			randdist = torch.rand(x.shape,device=x.device)
			block_pattern = ((1 -valid_block_center + float(1 - seed_drop_rate) + randdist) >= 1).float()
		
			if self.block_size == width and self.block_size == height:            
				block_pattern = torch.min(block_pattern.view(x.size(0),x.size(1),x.size(2)*x.size(3)),dim=2)[0].unsqueeze(-1).unsqueeze(-1)
			else:
				block_pattern = -F.max_pool2d(input=-block_pattern, kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)

			if self.block_size % 2 == 0:
					block_pattern = block_pattern[:, :, :-1, :-1]
			percent_ones = block_pattern.sum() / float(block_pattern.numel())

			if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
				wtsize=self.weight_behind.size(3)
				weight_max=self.weight_behind.max(dim=0,keepdim=True)[0]
				sig=torch.ones(weight_max.size(),device=weight_max.device)
				sig[torch.rand(weight_max.size(),device=sig.device)<0.5]=-1
				weight_max=weight_max*sig 
				weight_mean=weight_max.mean(dim=(2,3),keepdim=True)
				if wtsize==1:
					weight_mean=0.1*weight_mean
				#print(weight_mean)
			mean=torch.mean(x).clone().detach()
			var=torch.var(x).clone().detach()

			if not (self.weight_behind is None) and not(len(self.weight_behind)==0):
				dist=self.alpha*weight_mean*(var**0.5)*torch.randn(*x.shape,device=x.device)
			else:
				dist=self.alpha*0.01*(var**0.5)*torch.randn(*x.shape,device=x.device)

		x=x*block_pattern
		dist=dist*(1-block_pattern)
		x=x+dist
		x=x/percent_ones
		return x