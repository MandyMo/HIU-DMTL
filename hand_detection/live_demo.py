
import os
import cv2
from os import path as osp
import sys
import numpy as np
import torch
from model import Model
from utils import copy_state_dict, _decode_box, _nms
from copy import deepcopy

cur_dir = osp.dirname(osp.abspath(__file__))

class Demo(object):
	def __init__(self, chkp_path):
		self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
		self._create_model_(chkp_path=chkp_path)

	def _create_model_(self, chkp_path):
		model = Model(s_backbone='resnet18')
		if osp.isfile(chkp_path):
			pre_state_dict = torch.load(chkp_path, map_location='cpu')
			copy_state_dict(model.state_dict(), pre_state_dict, prefix='module.')

		self.model = model.to(self.device).eval()

	def pad_img(self, I):
		h, w, c = I.shape
		if h > w:
			ty, by = 0, 0
			lx, rx = (h-w)//2, (h-w+1)//2
		else: #w > h
			ty, by = (w-h)//2, (w-h+1)//2
			lx, rx = 0, 0
		return np.pad(I, ((ty, by), (lx, rx), (0, 0)), mode='constant')

	def vis_hms(self, p_hms, h, w):
		p_hms = cv2.resize(p_hms[0].cpu().numpy().transpose(1,2,0), (max(h, w), max(h, w)), interpolation=cv2.INTER_CUBIC)[...,np.newaxis]
		p_hms = (p_hms.clip(0, 1)*255.0).repeat(3, 2).astype(np.uint8)
		y, x = (0, (h-w)//2) if h > w else ((w-h)//2, 0)
		return p_hms[y:y+h, x:x+w].copy()

	def vis_whs(self, p_whs, h, w):
		p_whs = cv2.resize(p_whs[0].cpu().numpy().transpose(1,2,0), (max(h, w), max(h, w)), interpolation=cv2.INTER_NEAREST)*255.0
		p_pad = np.zeros((max(h, w), max(h, w), 1))
		p_whs = np.concatenate([p_whs, p_pad], 2).astype(np.uint8)
		x, y = (0, (h-w)//2) if h > w else ((w-h)//2, 0)
		return p_whs[y:y+h, x:x+w].copy()

	def __call__(self, I):
		h, w, c = I.shape
		img = torch.tensor(cv2.resize(self.pad_img(I), (256, 256), interpolation=cv2.INTER_CUBIC).transpose(2,0,1)[np.newaxis]).to(self.device)/127.5-1.0
		off = np.array([h-w, 0])/2.0 if h > w else np.array([0, w-h])/2
		p_hms, p_twths, p_txtys = torch.split(self.model(img), (1, 2, 2), dim=1)
		p_box = _decode_box(scores=deepcopy(p_hms), twths=deepcopy(p_twths), dxdys=deepcopy(p_txtys), k=10, threshold=0.5)[0]
		
		if len(p_box[0]) > 0:
			scores, lts, rbs = p_box[0].detach().cpu().numpy(), p_box[3].detach().cpu().numpy(), p_box[4].detach().cpu().numpy()
			keeps = _nms(dets=np.concatenate([lts, rbs],1), scores=scores, nms_thresh=0.7)

			for (lt, rb) in zip(p_box[3][keeps], p_box[4][keeps]):
				lt, rb = lt.detach().cpu().numpy().astype(np.float), rb.detach().cpu().numpy().astype(np.float)
				lt, rb = (lt * max(h, w) - off).astype(np.int), (rb * max(h,w) - off).astype(np.int)
				cv2.rectangle(I, tuple(lt), tuple(rb), (255,0,0), 2)

		p_hms = self.vis_hms(p_hms=p_hms,   h=h, w=w)
		p_whs = self.vis_whs(p_whs=p_twths, h=h, w=w)

		return np.concatenate([I, p_hms], 1).copy()


def main():
	demo_a = Demo(chkp_path=osp.join(cur_dir, 'checkpoint', 'hand_det.pth'))
	cap = cv2.VideoCapture(0)

	torch.set_grad_enabled(False)
	while True:
		_, frame = cap.read()
		h,w,c = frame.shape
		frame = demo_a(frame[:,::-1].copy())
		cv2.imshow('frame', frame)

		if cv2.waitKey(5) == 113:
			break





if __name__ == '__main__':
	main()