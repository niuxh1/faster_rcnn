import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


def loc_new_box(src_box, loc):
    if src_box.shape[0] == 0:
        return torch.zeros((0, 4), device=loc.device)
    src_width = torch.unsqueeze(src_box[:, 2] - src_box[:, 0], dim=-1)
    src_height = torch.unsqueeze(src_box[:, 3] - src_box[:, 1], dim=-1)
    src_ctr_x = torch.unsqueeze(src_box[:, 0], dim=-1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_box[:, 1], dim=-1) + 0.5 * src_height
    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    result_box = torch.zeros_like(loc)
    result_box[:, 0::4] = ctr_x - 0.5 * w
    result_box[:, 1::4] = ctr_y - 0.5 * h
    result_box[:, 2::4] = ctr_x + 0.5 * w
    result_box[:, 3::4] = ctr_y + 0.5 * h

    return result_box


class decode_box():
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes + 1

    def faster_RCNN_right_boxes(self, xy, wh, input_shape, image_shape):
        yx = xy[..., ::-1]
        hw = wh[..., ::-1]

        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        mins = yx - hw / 2.
        maxs = yx + hw / 2.

        boxes = np.concatenate([mins[..., 0:1], mins[..., 1:2], maxs[..., 0:1], maxs[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou=0.3, confidence=0.5):
        result = []
        """
        roi_cls_locs: (b,num_anchor,4)
        roi_scores: (b,num_anchor,num_classes)
        """
        bs = len(roi_cls_locs)

        rois = rois.view(bs, -1, 4)

        for i in range(bs):
            roi_cls_loc = roi_cls_locs[i] * self.std
            roi_cls_loc = roi_cls_loc.view(-1, self.num_classes, 4)
            roi = roi[i].view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_box = loc_new_box(roi.contiguous().view(-1, 4), roi_cls_loc.contiguous().view(-1, 4))
            cls_box = cls_box.view(-1, self.num_classes, 4)
            cls_box[..., [0, 2]] = cls_box[..., [0, 2]] / input_shape[1]
            cls_box[..., [1, 3]] = cls_box[..., [1, 3]] / input_shape[0]

            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            result.append([])

            for c in range(1, self.num_classes):

                c_conf = prob[:, c]
                c_conf_mask = c_conf > confidence

                if len(c_conf[c_conf_mask]) > 0:
                    right_box = cls_box[c_conf_mask, c]
                    right_score = c_conf[c_conf_mask]

                    keep = nms(right_box, right_score, nms_iou)

                    good_box = right_box[keep]
                    good_score = right_score[keep].unsqueeze(-1)
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda()

                    pred = torch.cat([good_box, good_score, labels], dim=-1).cpu().numpy()

                    result[-1].extend(pred)

            if len(result[-1]) > 0:
                result[-1] = np.array(result[-1])
                box_xy, box_wh = (result[-1][:, 0:2] + result[-1][:, 2:4]) / 2, result[-1][:, 2:4] - result[-1][:, 0:2]
                result[-1][:,0:4] = self.faster_RCNN_right_boxes(box_xy, box_wh, input_shape, image_shape)

        return result
