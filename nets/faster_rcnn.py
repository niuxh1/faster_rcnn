import torch
import torch.nn as nn

from nets.classifier import vgg16_roi_head
from nets.vgg16 import load_vgg16
from nets.rpns import region_proposal_nets


class Faster_RCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 pretrained=False):
        super().__init__()

        self.feat_stride = feat_stride

        self.extractor, classifier = load_vgg16(pretrained)
        self.rpns = region_proposal_nets(
            512,
            512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride,
            mode=mode
        )
        self.head = vgg16_roi_head(
            n_classes=num_classes + 1,
            roi_size=7,
            spatial_scale=1,
            classifier=classifier
        )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":

            image_szie = x.shape[2:]

            base_feature = self.extractor.forward(x)
            _, _, rois, rois_indices, _ = self.rpns.forward(base_feature, image_szie, scale)

            rois_cls_locs, rois_scores = self.head.forward(base_feature, rois, rois_indices, image_szie)

            return rois_cls_locs, rois_scores, rois, rois_indices
        elif mode == "extractor":

            base_feature = self.extractor.forward(x)

            return base_feature

        elif mode == "rpn":

            base_feature, img_size = x
            rpn_locs, rpn_scores, rois, rois_indices, anchors= self.rpns.forward(base_feature, img_size, scale)

            print("rpn_locs_type:", type(rpn_locs))
            print("rpn_locs_shape:", rpn_locs.shape)

            return rpn_locs, rpn_scores, rois, rois_indices, anchors

        elif mode == "head":

            base_feature, rois, rois_indices, img_size = x
            rois_cls_locs, rois_scores = self.head.forward(base_feature, rois, rois_indices, img_size)

            return rois_cls_locs, rois_scores

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()




