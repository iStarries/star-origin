import numpy as np
import torch
import torch.nn.functional as F


class BoundaryMetric:
    """Compute Boundary IoU and Boundary F-score for a class set."""

    def __init__(self, target_classes, boundary_width=3, min_mask_pixels=50):
        self.target_classes = sorted(set(int(c) for c in target_classes))
        self.boundary_width = max(1, int(boundary_width))
        self.min_mask_pixels = int(min_mask_pixels)

        self.biou_per_class = {cls: [] for cls in self.target_classes}
        self.bf_per_class = {cls: [] for cls in self.target_classes}

    def add_batch(self, gt_batch, pred_batch):
        if gt_batch.shape != pred_batch.shape:
            raise ValueError("GT and prediction must share the same shape.")

        for gt, pred in zip(gt_batch, pred_batch):
            self._add_single(gt, pred)

    def _add_single(self, gt, pred):
        for cls_idx in self.target_classes:
            gt_mask = gt == cls_idx
            if gt_mask.sum() < self.min_mask_pixels:
                continue

            pred_mask = pred == cls_idx
            boundary_gt = self._thick_boundary(gt_mask)
            boundary_pred = self._thick_boundary(pred_mask)

            union = np.logical_or(boundary_gt, boundary_pred)
            if union.sum() > 0:
                intersection = np.logical_and(boundary_gt, boundary_pred)
                self.biou_per_class[cls_idx].append(intersection.sum() / union.sum())

            bf = self._bf_score(boundary_gt, boundary_pred)
            if bf is not None:
                self.bf_per_class[cls_idx].append(bf)

    def _bf_score(self, boundary_gt, boundary_pred):
        if boundary_gt.sum() == 0 and boundary_pred.sum() == 0:
            return None

        dilated_gt = self._dilate(boundary_gt)
        dilated_pred = self._dilate(boundary_pred)

        precision_denom = boundary_pred.sum()
        recall_denom = boundary_gt.sum()

        precision = None if precision_denom == 0 else (np.logical_and(boundary_pred, dilated_gt).sum() / precision_denom)
        recall = None if recall_denom == 0 else (np.logical_and(boundary_gt, dilated_pred).sum() / recall_denom)

        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def _thick_boundary(self, mask):
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)

        if mask.sum() == 0:
            return np.zeros_like(mask, dtype=bool)

        tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        kernel_size = 2 * self.boundary_width + 1
        padding = self.boundary_width

        dilated = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-tensor, kernel_size=kernel_size, stride=1, padding=padding)
        boundary = (dilated - eroded).clamp(min=0)

        return (boundary > 0).squeeze(0).squeeze(0).cpu().numpy().astype(bool)

    def _dilate(self, mask):
        tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        kernel_size = 2 * self.boundary_width + 1
        padding = self.boundary_width
        dilated = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=padding)
        return (dilated > 0).squeeze(0).squeeze(0).cpu().numpy().astype(bool)

    def summary(self):
        biou_cls = {cls: (float(np.mean(vals)) if len(vals) > 0 else float("nan"))
                    for cls, vals in self.biou_per_class.items()}
        bf_cls = {cls: (float(np.mean(vals)) if len(vals) > 0 else float("nan"))
                  for cls, vals in self.bf_per_class.items()}

        biou_macro = float(np.nanmean(list(biou_cls.values()))) if biou_cls else float("nan")
        bf_macro = float(np.nanmean(list(bf_cls.values()))) if bf_cls else float("nan")

        return {
            "biou_per_class": biou_cls,
            "bf_per_class": bf_cls,
            "biou_old": biou_macro,
            "bf_old": bf_macro,
        }
