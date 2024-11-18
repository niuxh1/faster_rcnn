import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from utils.anchors import generate_anchor_base, _enumerate_shifted_anchor
from utils.bbox import loc_new_box


class proposal_creator():
    def __init__(self, mode, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=600,
                 n_test_pre_nms=3000, n_test_post_nms=300, min_size=16):

        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, locations, score, anchor, img_size, scale=1):

        if self.mode == 'training':
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor).type_as(locations)

        roi = loc_new_box(anchor, locations)
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        min_size = self.min_size * scale
        keep = torch.where((roi[:, 2] - roi[:, 0] >= min_size) & (roi[:, 3] - roi[:, 1] >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        ordered_score = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            ordered_score = ordered_score[:n_pre_nms]
        roi = roi[ordered_score, :]
        score = score[ordered_score]

        keep = nms(roi, score, self.nms_thresh)
        if len(keep) < n_post_nms:
            extra = np.random.choice(range(len(keep)), size=n_post_nms - len(keep), replace=True)
            keep = torch.cat([keep, keep[extra]])

        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class region_proposal_nets(nn.Module):
    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 feat_stride=16,
                 mode="training"
                 ):
        super().__init__()

        self.anchor_base = generate_anchor_base(ratios=ratios, anchor_scales=anchor_scales)
        anchor_num = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        self.score = nn.Conv2d(mid_channels, anchor_num * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, anchor_num * 4, 1, 1, 0)

        self.feature_stride = feat_stride

        self.proposal_layer = proposal_creator(mode)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1):
        n,_,h,w=x.shape
        x=F.relu(self.conv1(x))

        rpn_locs=self.loc(x)
        rpn_locs=rpn_locs.permute(0,2,3,1).contiguous().view(n,-1,4)

        rpn_scores=self.score(x)
        rpn_scores=rpn_scores.permute(0,2,3,1).contiguous().view(n,-1,2)

        scores_softmax=F.softmax(rpn_scores,dim=-1)
        rpn_fg_scores=scores_softmax[:,:,1]
        rpn_fg_scores=rpn_fg_scores.view(n,-1)

        anchor=_enumerate_shifted_anchor(np.array(self.anchor_base),self.feature_stride,h,w)
        rois=list()
        rois_indices=list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale)
            batch_index=i*torch.ones((len(roi),))
            rois.append(roi)
            rois_indices.append(batch_index.unsqueeze(0))

        rois=torch.cat(rois,dim=0).type_as(x)
        rois_indices=torch.cat(rois_indices,dim=0).type_as(x)
        anchor=torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)



        return rpn_locs,rpn_scores,rois,rois_indices,anchor

def normal_init(x, std, mean, truncated=False):
    if truncated:
        x.weight.data.normal_().fmod_(2).mul_(std).add_(mean)
    else:
        x.weight.data.normal_(mean, std)
        x.bias.data.zero_()
