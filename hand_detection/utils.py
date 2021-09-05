

import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

#scores ： b x c x h x w
def _topk(scores, k):
    B, C, H, W = scores.size()
    
    topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), k)

    topk_inds = topk_inds % (H * W)
    
    topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), k)
    # topk_clses = torch.div(topk_ind, k, rounding_mode='floor')
    topk_clses = (topk_ind / (H * W)).int()
    topk_inds = _gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, k)

    return topk_score, topk_inds, topk_clses

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

#scores: n x c x h x w
#twths： n x 2 x h x w
def _decode_box(scores, twths, dxdys, k=2, threshold=0.8, stride=4, kernel=5):
    N, C, H, W = scores.shape
    boxs = []
    scores_max = F.max_pool2d(scores, kernel_size=kernel, padding=(kernel-1)//2, stride=1)
    keep = (scores_max == scores).float()
    scores *= keep
    topk_scores, topk_inds, topk_clses = _topk(scores=scores, k=k)
    
    for (topk_score, topk_ind, topk_cls, twth, dxdy) in zip(topk_scores, topk_inds, topk_clses, twths, dxdys):
        mask = topk_score >= threshold
        topk_score, topk_ind, topk_cls = topk_score[mask], topk_ind[mask], topk_cls[mask]
        if mask.sum() == 0:
            boxs.append([[], [], [], [], []])
            continue
        twth = twth.reshape(2, -1)[:, topk_ind].permute(1, 0)/2.0
        dwdh = dxdy.reshape(2, -1)[:, topk_ind].permute(1, 0)
        w,   h = (topk_ind%W).float()/W, (topk_ind//W).float()/H
        wh = torch.stack([w,h], -1).float() + dwdh
        boxs.append([topk_score, topk_ind, topk_cls, wh-twth, wh+twth])
    return boxs

def deterministic_training_procedure(seed=2020):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def copy_state_dict(cur_state_dict, pre_state_dict, prefix=''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            try:
                cur_state_dict[k].copy_(v)
            except:
                cur_state_dict[k].copy_(v.view(cur_state_dict[k].shape))
        except:
            print('copy param {} failed'.format(k))
            continue

class AverageMeter(object):

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.initialized = True

    def update(self, val, weight=0.01):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = (1 - weight) * self.val + weight * val
        self.avg = self.val

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def reset(self):
        self.initialized = False



def _nms(dets, scores, nms_thresh=0.7):
    """"Pure Python NMS baseline."""
    x1 = dets[:, 0]  #xmin
    y1 = dets[:, 1]  #ymin
    x2 = dets[:, 2]  #xmax
    y2 = dets[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)                
    order = scores.argsort()[::-1]                       

    keep = []              
    while order.size > 0:
        i = order[0]                                     
        keep.append(i)                                  
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def _ious(gt_boxs, pred_boxs):
    ious = []
    pred_boxs = pred_boxs.reshape(-1, 2, 2)
    for gt_box in gt_boxs:
        gt_box = gt_box.reshape(1, 2, 2)
        lt, rb = np.maximum(gt_box[:, 0], pred_boxs[:, 0]), np.minimum(gt_box[:, 1], pred_boxs[:, 1])
        w,  h  = np.maximum(rb[:, 0] - lt[:, 0], 0), np.maximum(rb[:, 1] - lt[:, 1], 0)
        inter  = w * h
        over   = np.maximum((pred_boxs[:, 1, 0] - pred_boxs[:, 0, 0]) * (pred_boxs[:, 1, 1] - pred_boxs[:, 0, 1]), 1e-6) +\
            (gt_box[0, 1, 0] - gt_box[0, 0, 0])*(gt_box[0, 1, 1] - gt_box[0, 1, 0]) - inter
        
        ious.append((inter/over).max())

    return ious

def _batch_iou(b_gt_boxs, b_pred_boxs):
    ious = []
    for (gt_boxs, pred_boxs) in zip(b_gt_boxs, b_pred_boxs):
        ious += _ious(gt_boxs, pred_boxs)
    return np.mean(ious)