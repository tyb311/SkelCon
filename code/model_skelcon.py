
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):
		super(BasicConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#BasicConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()

		self.o = nn.Identity()
		drop_prob=0.15
		self.o = nn.Dropout2d(p=drop_prob, inplace=False) 

		if activation is None:
			self.a = nn.Identity()
		else:
			self.a = activation

	def forward(self, x):
		x = self.c(x)# x = torch.clamp_max(x, max=99)
		# print('c:', x.max().item())
		x = self.o(x)
		# print('o:', x.max().item())
		x = self.b(x)
		# print('b:', x.max().item())
		x = self.a(x)
		# print('a:', x.max().item())
		return x


class Bottleneck(nn.Module):
	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, downsample=None, **args):
		super(Bottleneck, self).__init__()

		self.conv1 = self.MyConv(in_c, out_c, kernel_size=3, stride=stride)
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(out_c)
		)
		self.relu = swish#nn.LeakyReLU()
		if downsample is None and in_c != out_c:
			downsample = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
				nn.BatchNorm2d(out_c),
			)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		if self.downsample is not None:
			residual = self.downsample(x)
		# print('Basic:', x.shape)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.relu(out + residual)
		# print(out.min().item(), out.max().item())
		return out

class ConvBlock(torch.nn.Module):
	attention=None
	MyConv = BasicConv2d
	def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
		super(ConvBlock, self).__init__()
		self.shortcut = nn.Sequential(nn.Conv2d(inp_c, out_c, kernel_size=1), nn.BatchNorm2d(out_c))
		pad = (ksize - 1) // 2

		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		block = []
		block.append(self.MyConv(inp_c, out_c, kernel_size=ksize, padding=pad))

		block.append(self.MyConv(out_c, out_c, kernel_size=ksize, padding=pad))
		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return swish(out + self.shortcut(x))

# 输出层 & 下采样
class OutSigmoid(nn.Module):
	def __init__(self, inp_planes, out_planes=1, out_c=8):
		super(OutSigmoid, self).__init__()
		self.cls = nn.Sequential(
			nn.Conv2d(in_channels=inp_planes, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			# nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_c),
			nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1, bias=True),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.cls(x)

class UpsampleBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, up_mode='transp_conv'):
		super(UpsampleBlock, self).__init__()
		block = []
		if up_mode == 'transp_conv':
			block.append(nn.ConvTranspose2d(inp_c, out_c, kernel_size=2, stride=2))
		elif up_mode == 'up_conv':
			block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
			block.append(nn.Conv2d(inp_c, out_c, kernel_size=1))
		else:
			raise Exception('Upsampling mode not supported')

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class ConvBridgeBlock(torch.nn.Module):
	def __init__(self, out_c, ksize=3):
		super(ConvBridgeBlock, self).__init__()
		pad = (ksize - 1) // 2
		block=[]

		block.append(nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=pad))
		block.append(nn.LeakyReLU())
		block.append(nn.BatchNorm2d(out_c))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class UpConvBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, ksize=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
		super(UpConvBlock, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = UpsampleBlock(inp_c, out_c, up_mode=up_mode)
		self.conv_layer = ConvBlock(2 * out_c, out_c, ksize=ksize, shortcut=shortcut, pool=False)
		if self.conv_bridge:
			self.conv_bridge_layer = ConvBridgeBlock(out_c, ksize=ksize)

	def forward(self, x, skip):
		up = self.up_layer(x)
		if self.conv_bridge:
			out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
		else:
			out = torch.cat([up, skip], dim=1)
		out = self.conv_layer(out)
		return out

class LUNet(nn.Module):
	__name__ = 'lunet'
	use_render = False
	def __init__(self, inp_c=1, n_classes=1, layers=(32,32,32,32,32), num_emb=128):
		super(LUNet, self).__init__()
		self.num_features = layers[-1]

		self.__name__ = 'u{}x{}'.format(len(layers), layers[0])
		self.n_classes = n_classes
		self.first = BasicConv2d(inp_c, layers[0])

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(inp_c=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(inp_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		self.conv_bn = nn.Sequential(
			nn.Conv2d(layers[0], layers[0], kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(layers[0]),
		)
		self.aux = nn.Sequential(
			nn.Conv2d(layers[0], n_classes, kernel_size=1),
			# nn.Conv2d(n_classes, n_classes, kernel_size=1),
			nn.BatchNorm2d(n_classes),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.first(x)
		down_activations = []
		for i, down in enumerate(self.down_path):
			down_activations.append(x)
			# print(x.shape)
			x = down(x)
		down_activations.reverse()

		for i, up in enumerate(self.up_path):
			x = up(x, down_activations[i])

		# self.feat = F.normalize(x, dim=1, p=2)
		x = self.conv_bn(x)
		self.feat = x

		self.pred = self.aux(x)
		return self.pred


def lunet(**args):
	ConvBlock.attention = None
	net = LUNet(**args)
	net.__name__ = 'lunet'
	return net


def similar_matrix2(q, k, temperature=0.1):#负太多了
	# print('similar_matrix2:', q.shape, k.shape)
	qfh, qfl, qbh, qbl = torch.chunk(q, 4, dim=0)
	kfh, kfl, kbh, kbl = torch.chunk(k, 4, dim=0)

	# negative logits: NxK
	l_pos = torch.einsum('nc,kc->nk', [qfl, kfh])
	l_neg = torch.einsum('nc,kc->nk', [qbl, kbh])
	# print(l_pos.shape, l_neg.shape)
	return 2 - l_pos.mean() - l_neg.mean()

CLLOSSES = {
	'sim2':similar_matrix2, 
	}

class SIAM(nn.Module):
	__name__ = 'siam'
	def __init__(self,
				 encoder,
				 clloss='nce',
				 temperature=0.1,
				 proj_num_layers=2,
				 pred_num_layers=2,
				 proj_num_length=64,
				 **kwargs):
		super().__init__()
		self.loss = CLLOSSES[clloss]
		self.encoder = encoder
		# self.__name__ = self.encoder.__name__
		self.__name__ = 'X'.join([self.__name__, self.encoder.__name__]) #, clloss
		
		self.temperature = temperature
		self.proj_num_layers = proj_num_layers
		self.pred_num_layers = pred_num_layers

		self.projector = self.encoder.projector
		self.predictor = self.encoder.predictor

	def forward(self, img, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		out = self.encoder(img, **args)
		self.pred = self.encoder.pred
		self.feat = self.encoder.feat
		# self._dequeue_and_enqueue(proj1_ng, proj2_ng)
		if hasattr(self.encoder, 'tmp'):
			self.tmp = self.encoder.tmp
		return out

	def constraint(self, **args):
		return self.encoder.constraint(**args)

	def regular(self, sampler, lab, fov=None):#contrastive loss split by classification
		feat = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		proj = self.projector(feat)
		self.proj = proj
		pred = self.predictor(proj)

		# compute loss
		losSG1 = self.loss(pred, proj.detach(), temperature=self.temperature)
		losSG2 = self.loss(proj, pred.detach(), temperature=self.temperature)
		return losSG1 + losSG2

class MlpNorm(nn.Module):
	def __init__(self, dim_inp=256, dim_out=64):
		super(MlpNorm, self).__init__()
		dim_mid = min(dim_inp, dim_out)#max(dim_inp, dim_out)//2
		# hidden layers
		linear_hidden = []#nn.Identity()
		# for i in range(num_layers - 1):
		linear_hidden.append(nn.Linear(dim_inp, dim_mid))
		linear_hidden.append(nn.Dropout(p=0.2))
		linear_hidden.append(nn.BatchNorm1d(dim_mid))
		linear_hidden.append(nn.LeakyReLU())
		self.linear_hidden = nn.Sequential(*linear_hidden)

		self.linear_out = nn.Linear(dim_mid, dim_out)# if num_layers >= 1 else nn.Identity()

	def forward(self, x):
		x = self.linear_hidden(x)
		x = self.linear_out(x)
		return F.normalize(x, p=2, dim=-1)

def torch_dilation(x, ksize=3, stride=1):
	return F.max_pool2d(x, (ksize, ksize), stride, ksize//2)

class MorphBlock(nn.Module):
	def __init__(self, inp_ch=2, channel=8):
		super().__init__()
		self.ch_wv = nn.Sequential(
			nn.Conv2d(inp_ch,channel,kernel_size=5, padding=2),
			nn.Conv2d(channel,channel,kernel_size=5, padding=2),
			nn.BatchNorm2d(channel),
			nn.Conv2d(channel,channel//2,kernel_size=3, padding=1),
		)
		self.ch_wq = nn.Sequential(
			nn.Conv2d(channel//2,8,kernel_size=3, padding=1),
			nn.BatchNorm2d(8),
			nn.Conv2d(8,1,kernel_size=1),
			nn.Sigmoid()
		)
	
	def forward(self, x, o):
		x = torch.cat([torch_dilation(o, ksize=3), x, o], dim=1)#, 1-torch_dilation(1-x, ksize=3)
		x = self.ch_wv(x)
		# print(x.shape)
		return self.ch_wq(x)

class SeqNet(nn.Module):#Supervised contrastive learning segmentation network
	__name__ = 'scls'
	def __init__(self, type_net, type_seg, num_emb=128):
		super(SeqNet, self).__init__()

		self.fcn = eval(type_net+'(num_emb=num_emb)')#build_model(cfg['net']['fcn'])
		self.seg = eval(type_seg+'(inp_c=32)')#build_model(cfg['net']['seg'])

		self.projector = MlpNorm(32, num_emb)#self.fcn.projector#MlpNorm(32, 64, num_emb)
		self.predictor = MlpNorm(num_emb, num_emb)#self.fcn.predictor#MlpNorm(32, 64, num_emb)

		self.morpholer1 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.morpholer2 = MorphBlock(32+2)#形态学模块使用一个还是两个哪？
		self.__name__ = '{}X{}'.format(self.fcn.__name__, self.seg.__name__)

	def constraint(self, aux=None, fun=None, **args):
		aux = torch_dilation(aux)
		los1 = fun(self.sdm1, aux)
		los2 = fun(self.sdm2, aux)
		# if self.__name__.__contains__('dmf'):
		# 	los1 = los1 + self.fcn.regular()*0.1
		return los1, los2
	
	def regular(self, sampler, lab, fov=None, return_loss=True):
		emb = sampler.select(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		emb = self.projector(emb)
		# print(emb.shape)
		self.emb = emb
		if return_loss:
			return sampler.infonce(emb)
	tmp = {}
	def forward(self, x):
		aux = self.fcn(x)
		self.feat = self.fcn.feat
		out = self.seg(self.feat)
		self.pred = out
		# print(self.fcn.feat.shape, self.seg.feat.shape)
		self.sdm1 = self.morpholer1(self.fcn.feat, aux)
		self.sdm2 = self.morpholer2(self.seg.feat, out)
		self.tmp = {'sdm1':self.sdm1, 'sdm2':self.sdm2}

		if self.training:
			if isinstance(aux, (tuple, list)):
				return [self.pred, aux[0], aux[1]]
			else:
				return [self.pred, aux]
		return self.pred

def build_model(type_net='', type_seg='', type_loss='sim2', type_arch='', num_emb=128):
	model = eval(type_net+'(num_emb=num_emb)')

	model = SeqNet(type_net, type_seg, num_emb=num_emb)
	model = SIAM(encoder=model, clloss=type_loss, proj_num_length=num_emb)
	model.__name__ = model.__name__.replace('siamXlunetXlunet', 'SLL')

	return model