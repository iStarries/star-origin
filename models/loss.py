import torch
import torch.nn as nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    """BCE loss that optionally injects teacher soft targets for old classes.

    When ``logit_old`` is provided, the target tensor is built by:
    1. Filling the old-class channels with the teacher's sigmoid probability
       (``logit_old.sigmoid()``), either on **all pixels** or only on
       background pixels depending on ``distill_bg_only``.
    2. Writing the current ground-truth one-hot labels into the new-class
       channels.

    """

    def __init__(self, ignore_index=255, ignore_bg=True, pos_weight=None, reduction='mean', distill_bg_only=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.distill_bg_only = distill_bg_only

        if ignore_bg is True:
            self.ignore_indexes = [0, self.ignore_index]
        else:
            self.ignore_indexes = [self.ignore_index]

    def forward(self, logit, label, logit_old=None):
        # logit:     [N, C_tot, H, W]
        # logit_old: [N, C_prev, H, W]
        # label:     [N, H, W] or [N, C, H, W]
        C = logit.shape[1]
        valid_labels = label[label != self.ignore_index]
        max_label = valid_labels.max().item() if valid_labels.numel() > 0 else -1
        # If logits do not include background channel (C == max_label), shift
        # semantic class ids by 1 so class k maps to channel k-1.
        channel_offset = 1 if max_label >= C else 0
        if logit_old is None:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    channel_idx = int(cls_idx) - channel_offset
                    if 0 <= channel_idx < C:
                        target[:, channel_idx] = (label == int(cls_idx)).float()
            elif len(label.shape) == 4:
                target = label
            else:
                raise NotImplementedError
            
            logit = logit.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.permute(0, 2, 3, 1).reshape(-1, C)

            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
        else:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)

                teacher_prob = logit_old.sigmoid()
                if self.distill_bg_only:
                    bg_mask = (label == 0).unsqueeze(1)  # [N,1,H,W]
                    teacher_prob = teacher_prob * bg_mask

                target[:, :logit_old.shape[1]] = teacher_prob

                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    channel_idx = int(cls_idx) - channel_offset
                    if 0 <= channel_idx < C:
                        target[:, channel_idx] = (label == int(cls_idx)).float()
            else:
                raise NotImplementedError
            
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
            del target

            return loss


class WBCELoss(nn.Module):
    def __init__(self, ignore_index=255, pos_weight=None, reduction='none', n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:train_voc.sh-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes  # |Ct|, 19-1: 1 | 15-5: 5 | 15-1: 1
        
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)
        
    def forward(self, logit, label):
        # logit:     [N, |Ct|, H, W]
        # label:     [N, H, W]

        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()
        # for cls_idx in label.unique():
        for cls_idx in torch.clamp(label, min=0).unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()
            # target[:, int(cls_idx) - 1] = (label == int(cls_idx)).float()
        
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )

        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        elif self.reduction == 'mean':
            return loss
        else:
            raise NotImplementedError


class PKDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, features, features_old, pseudo_label_region):

        pseudo_label_region_5 = F.interpolate(
            pseudo_label_region, size=features[2].shape[2:], mode="bilinear", align_corners=False)

        loss_5 = self.criterion(features[5], features_old[5])
        loss_5 = (loss_5 * pseudo_label_region_5).sum() / (pseudo_label_region_5.sum() * features[5].shape[1])

        return loss_5


class ContLoss(nn.Module):
    def __init__(self, ignore_index=255, n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:train_voc.sh-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes

        self.criteria = nn.MSELoss(reduction='mean')

    def forward(self, features, logit, label, prev_prototypes=None):

        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()
        for cls_idx in label.unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()

        small_target = F.interpolate(target, size=features.shape[2:], mode='bilinear', align_corners=False)
        new_center = F.normalize(features, p=2, dim=1).unsqueeze(1) * small_target.unsqueeze(2)
        new_center = F.normalize(new_center.sum(dim=[0, 3, 4]), p=2, dim=1)

        dist_pp = torch.norm(new_center.unsqueeze(0) - prev_prototypes.unsqueeze(1), p=2, dim=2)
        l_neg = (1 / dist_pp.min(0).values).mean()

        return l_neg