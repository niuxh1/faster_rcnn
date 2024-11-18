import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool


class vgg16_roi_head(nn.Module):
    def __init__(self, n_classes, roi_size, spatial_scale, classifier):
        super().__init__()
        self.classifier = classifier
        self.bbox = nn.Linear(4096, n_classes * 4)
        self.score = nn.Linear(4096, n_classes)

        normal_init(self.bbox, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape

        roi_indices = roi_indices.cuda()
        rois = rois.cuda()

        roi_indices=roi_indices.unsqueeze(0)

        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        roi_features = torch.zeros_like(rois)
        roi_features[:, [0, 2]] = rois[:, [0, 2]] / img_size[0] * x.size()[3]
        roi_features[:, [1, 3]] = rois[:, [1, 3]] / img_size[1] * x.size()[2]

        indices_rois = torch.cat([roi_indices.unsqueeze(-1), roi_features], dim=1)

        pool = self.roi(x, indices_rois)

        pool = pool.view(pool.size(0), -1)

        fc_cls = self.classifier(pool)

        roi_cls = self.score(fc_cls)
        roi_bbox = self.bbox(fc_cls)

        roi_cls = roi_cls.view(n, -1, roi_cls.size(1))
        roi_bbox = roi_bbox.view(n, -1, roi_bbox.size(1))

        return roi_cls, roi_bbox


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
