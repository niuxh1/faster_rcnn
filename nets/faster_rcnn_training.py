import math
from functools import partial

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def box_iou(box_a, box_b):
    if box_a.shape[1] != 4 or box_b.shape[1] != 4:
        raise IndexError
    tl = np.maximum(box_a[:, None, :2], box_b[:, :2])
    br = np.minimum(box_a[:, None, 2:], box_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(box_a[:, 2:] - box_a[:, :2], axis=1)
    area_b = np.prod(box_b[:, 2:] - box_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


class anchor_target_creator():

    def __init__(self, num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.num_sample = num_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, box, anchor):
        argmax_iou, label = self._create_label(anchor, box)
        if (label > 0).any():
            loc = bbox2loc(anchor, box[argmax_iou])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, box):

        ious = box_iou(anchor, box)
        if len(box) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(box))

        argmax_ious = ious.argmax(axis=1)

        max_ious = np.max(ious, axis=1)

        target_argmax_ious = ious.argmax(axis=0)

        for i in range(len(target_argmax_ious)):
            argmax_ious[target_argmax_ious[i]] = i

        return argmax_ious, max_ious, target_argmax_ious

    def _create_label(self, anchor, box):

        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, target_argmax_ious = self._calc_ious(anchor, box)

        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1

        if len(target_argmax_ious) > 0:
            label[target_argmax_ious] = 1

        num_pos = int(self.pos_ratio * self.num_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > num_pos:
            give_up_index = np.random.choice(pos_index, size=(len(pos_index) - num_pos), replace=False)
            label[give_up_index] = -1

        num_neg = self.num_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > num_neg:
            give_up_index = np.random.choice(neg_index, size=(len(neg_index) - num_neg), replace=False)
            label[give_up_index] = -1

        return argmax_ious, label


class proposal_target_creator:
    def __init__(self, n_sample=256,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lw=0.0):

        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lw = neg_iou_thresh_lw
        self.pos_roi_image = np.round(n_sample * pos_ratio)

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        roi = roi.unsqueeze(0)

        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # ----------------------------------------------------- #
        #   计算建议框和真实框的重合程度
        # ----------------------------------------------------- #
        iou = box_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # ---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框  [num_roi, ]
            # ---------------------------------------------------------#
            gt_assignment = iou.argmax(axis=1)
            # ---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            # ---------------------------------------------------------#
            max_iou = iou.max(axis=1)
            # ---------------------------------------------------------#
            #   真实框的标签要+1因为有背景的存在
            # ---------------------------------------------------------#
            gt_roi_label = label[gt_assignment] + 1

        # ----------------------------------------------------------------#
        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        # ----------------------------------------------------------------#
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # -----------------------------------------------------------------------------------------------------#
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        # -----------------------------------------------------------------------------------------------------#
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lw))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # ---------------------------------------------------------#
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        # ---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label


class Faster_RCNN_training(nn.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1
        self.anchor_target_creator = anchor_target_creator()
        self.proposal_target_creator = proposal_target_creator()
        self.std = [0.1, 0.1, 0.2, 0.2]

    def _loc_loss(self, pred_loc, target_loc, target_label, sigma):
        pred_loc = pred_loc[target_label > 0]
        target_loc = target_loc[target_label > 0]

        sigma = sigma ** 2

        regression_diff = (target_loc - pred_loc).abs().float()

        regression_loss = torch.where(
            regression_diff < 1 / sigma,
            0.5 * sigma * regression_diff ** 2,
            regression_diff - 0.5 / sigma
        )

        regression_loss = regression_loss.sum()
        num_pos = (target_label > 0).float().sum()
        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))

        return regression_loss

    def forward(self, images, box, labels, scales):
        n = images.shape[0]
        image_size = images.shape[2:]

        base_feature = self.model(images, mode="extractor")

        rpn_locs, rpn_scores, rois, rois_indices, anchors = self.model(x=[base_feature, image_size], mode="rpn")

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        samples_rois, samples_indexs, target_rois_locs, target_rois_labels = [], [], [], []
        for i in range(n):
            the_box = box[i]
            the_label = labels[i]
            the_rpn_loc = rpn_locs[i]
            the_rpn_score = rpn_scores[i]
            the_roi = rois[i]

            traget_rpn_loc, target_rpn_label = self.anchor_target_creator(the_box, anchors[0].cpu().numpy())
            target_rpn_loc = torch.Tensor(traget_rpn_loc).type_as(rpn_locs)
            target_rpn_label = torch.Tensor(target_rpn_label).type_as(rpn_locs).long()

            rpn_loc_loss = self._loc_loss(the_rpn_loc, target_rpn_loc, target_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(the_rpn_score, target_rpn_label, ignore_index=-1)

            rpn_cls_loss_all += rpn_cls_loss
            rpn_loc_loss_all += rpn_loc_loss

            sample_roi, target_roi, target_roi_label = self.proposal_target_creator(the_roi, the_box, the_label,
                                                                                    self.std)
            samples_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            samples_indexs.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * rois_indices[i][0])
            target_rois_locs.append(torch.Tensor(target_roi).type_as(rpn_locs))
            target_rois_labels.append(torch.Tensor(target_roi_label).type_as(rpn_locs).long())

        samples_rois = torch.cat(samples_rois, dim=0)
        samples_indexs = torch.cat(samples_indexs, dim=0)
        roi_cls_locs, roi_scores = self.model(x=[base_feature, samples_rois, samples_indexs, image_size], mode="head")

        for i in range(n):
            num_sample = roi_cls_locs.size(0)

            the_cls_loc = roi_cls_locs[i]
            the_score = roi_scores[i]
            target_roi_loc = target_rois_locs[i]
            target_roi_label = target_rois_labels[i]

            the_cls_loc = the_cls_loc.view(num_sample, -1, 4)
            roi_loc = the_cls_loc[torch.arange(0, num_sample), target_roi_label]

            roi_loc_loss = self._loc_loss(roi_loc, target_roi_loc, target_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(the_score, target_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
        losses = losses + [sum(losses)]
        return losses

    def train_step(self, images, boxes, labels, scales, fp16, scaler=None):
        self.optimizer.zero_grad()

        if not fp16:
            losses = self.forward(images, boxes, labels, scales)
            losses[-1].backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                losses = self.forward(images, boxes, labels, scales)

            scaler.scale(losses[-1]).backward()
            scaler.step(self.optimizer)
            scaler.update()
        return losses


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# just copy
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
