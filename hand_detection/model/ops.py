
import sys
from os import path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(c_in, c_out, k, s, p, g=1, relu=True, bn=True, bias=False):
	ops = [nn.Conv2d(c_in, c_out, k, s, padding=p, groups=g, bias=bias)]
	if bn:
		ops.append(nn.BatchNorm2d(c_out))
	if relu:
		ops.append(nn.ReLU(inplace=True))
	return nn.Sequential(*ops)

def linear_layers(feats):
	ops, nl = [], len(feats)
	for _ in range(nl-2):
		ops.append(conv2d(feats[_], feats[_+1],bias=False, k=1,s=1,p=0,g=1,relu=True,bn=True))
	ops.append(conv2d(feats[nl-2],feats[nl-1], k=1,s=1,p=0,g=1,relu=False,bias=True,bn=False))

	return nn.Sequential(*ops)

class BottleNeck(nn.Module):
	def __init__(self, in_plane, out_plane, reduce_ratio=4, min_plane=128, s=1, residual=True):
		super(BottleNeck, self).__init__()
		mid_plane = max(in_plane//4, min_plane)
		self.convs = nn.Sequential(
			conv2d(c_in=in_plane, c_out=mid_plane,k=1,s=1,p=0,g=1,relu=True,bn=True,bias=False),
			conv2d(c_in=mid_plane,c_out=mid_plane,k=3,s=s,p=1,g=1,relu=True,bn=True,bias=False),
			conv2d(c_in=mid_plane,c_out=out_plane,k=1,s=1,p=0,g=1,relu=False,bn=True,bias=False),
		)
		
		if s!=1 and in_plane!=out_plane:
			self.residual=nn.Sequential(
				nn.AvgPool2d(kernel_size=s,stride=s),
				conv2d(c_in=in_plane,c_out=out_plane,k=1,s=1,p=0,g=1,relu=False,bn=True),
			)
		elif s!= 1:
			self.residual=nn.Sequential(
				nn.AvgPool2d(kernel_size=s,stride=s),
				nn.BatchNorm2d(out_plane),
			)
		elif in_plane!=out_plane:
			self.residual=conv2d(c_in=in_plane,c_out=out_plane,k=1,s=1,p=0,g=1,relu=False,bn=True)
		else:
			self.residual=nn.Sequential()

	def forward(self, x, *args, **kargs):
		out = self.convs(x)
		res = self.residual(x)
		return F.relu(out+res)