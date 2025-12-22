import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PhasePrototypeBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_shape: Tuple[int, int, int],
        r_low_ratio: float = 0.2,
        r_high_ratio: float = 0.6,
        use_cos_sin: bool = True,
        ema_beta: float = 0.01,
        amp_source: str = "ref",
        normalize_syn: bool = False,
        replay_detach_ref: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_shape = feat_shape
        self.r_low_ratio = r_low_ratio
        self.r_high_ratio = r_high_ratio
        self.use_cos_sin = use_cos_sin
        self.ema_beta = ema_beta
        self.amp_source = amp_source
        self.normalize_syn = normalize_syn
        self.replay_detach_ref = replay_detach_ref
        self.eps = eps

        c, h, w = feat_shape
        mid_mask = self.build_mid_mask(h, w, r_low_ratio, r_high_ratio, device="cpu")
        n_mid = int(mid_mask.sum().item())

        self.register_buffer("mid_mask", mid_mask)
        self.register_buffer("phase_cos", torch.zeros(num_classes, c, n_mid))
        self.register_buffer("phase_sin", torch.zeros(num_classes, c, n_mid))
        self.register_buffer("amp_mean", torch.zeros(num_classes, c, n_mid))
        self.register_buffer("count", torch.zeros(num_classes, dtype=torch.long))

    @staticmethod
    def build_mid_mask(
        height: int,
        width: int,
        r_low_ratio: float,
        r_high_ratio: float,
        device=None,
    ) -> torch.BoolTensor:
        center_h = (height - 1) / 2.0
        center_w = (width - 1) / 2.0
        y = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
        x = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
        dh = y - center_h
        dw = x - center_w
        r = torch.sqrt(dh ** 2 + dw ** 2)
        r_max = math.sqrt(center_h ** 2 + center_w ** 2)
        r_low = r_low_ratio * r_max
        r_high = r_high_ratio * r_max
        return (r_low <= r) & (r <= r_high)

    def is_ready(self, class_id: int) -> bool:
        if class_id < 0 or class_id >= self.num_classes:
            raise IndexError(
                f"[PHASE-PPB] class_id={class_id} out of range for num_classes={self.num_classes}. "
                f"This indicates mismatch between train_id mapping/head channels and PPB num_classes."
            )
        return int(self.count[class_id].item()) > 0

    def get_mid_size(self) -> int:
        return int(self.mid_mask.sum().item())

    def update_from_feature(
        self, feature: torch.Tensor, mask: torch.Tensor, class_id: int
    ) -> bool:
        if feature.ndim != 3:
            return False
        if class_id < 0 or class_id >= self.num_classes:
            raise IndexError(
                f"[PHASE-PPB] update_from_feature got invalid class_id={class_id} for num_classes={self.num_classes}. "
                f"Fix PPB expand/load or train_id mapping."
            )

        c, h, w = feature.shape
        if (c, h, w) != self.feat_shape:
            return False
        if mask.ndim != 2 or mask.shape != (h, w):
            return False
        if mask.sum() == 0:
            return False

        with torch.no_grad():
            feature_f = feature.float()
            mask_f = mask.to(feature_f.dtype).unsqueeze(0)
            feat_masked = feature_f * mask_f

            g = torch.fft.fft2(feat_masked)
            g = torch.fft.fftshift(g, dim=(-2, -1))
            amp = torch.abs(g)
            phase = torch.angle(g)

            amp_mid = amp[:, self.mid_mask]
            cos_mid = torch.cos(phase)[:, self.mid_mask]
            sin_mid = torch.sin(phase)[:, self.mid_mask]

            if int(self.count[class_id].item()) == 0:
                self.phase_cos[class_id] = cos_mid
                self.phase_sin[class_id] = sin_mid
                self.amp_mean[class_id] = amp_mid
            else:
                beta = self.ema_beta
                self.phase_cos[class_id] = (1 - beta) * self.phase_cos[class_id] + beta * cos_mid
                self.phase_sin[class_id] = (1 - beta) * self.phase_sin[class_id] + beta * sin_mid
                self.amp_mean[class_id] = (1 - beta) * self.amp_mean[class_id] + beta * amp_mid

            self.count[class_id] += 1

        return True

    def synthesize_from_reference(
        self, feature_ref: torch.Tensor, class_id: int
    ) -> Optional[torch.Tensor]:
        if feature_ref.ndim != 3:
            return None
        if class_id < 0 or class_id >= self.num_classes:
            raise IndexError(
                f"[PHASE-PPB] synthesize_from_reference got invalid class_id={class_id} for num_classes={self.num_classes}. "
                f"Fix PPB expand/load or train_id mapping."
            )
        c, h, w = feature_ref.shape
        if (c, h, w) != self.feat_shape:
            return None
        if not self.is_ready(class_id):
            return None

        if self.replay_detach_ref:
            feature_ref = feature_ref.detach()

        feature_dtype = feature_ref.dtype
        feature_ref = feature_ref.float()

        g_ref = torch.fft.fft2(feature_ref)
        g_ref = torch.fft.fftshift(g_ref, dim=(-2, -1))
        amp_ref = torch.abs(g_ref)
        phase_ref = torch.angle(g_ref)

        proto_phase = torch.atan2(self.phase_sin[class_id], self.phase_cos[class_id])
        phase_syn = phase_ref.clone()
        phase_syn[:, self.mid_mask] = proto_phase

        amp_syn = amp_ref.clone()
        if self.amp_source == "mean":
            amp_syn[:, self.mid_mask] = self.amp_mean[class_id]

        g_syn = torch.polar(amp_syn, phase_syn)
        g_syn = torch.fft.ifftshift(g_syn, dim=(-2, -1))
        x_syn = torch.fft.ifft2(g_syn).real

        if self.normalize_syn:
            ref_mean = feature_ref.mean(dim=(1, 2), keepdim=True)
            ref_std = feature_ref.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
            syn_mean = x_syn.mean(dim=(1, 2), keepdim=True)
            syn_std = x_syn.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
            x_syn = (x_syn - syn_mean) / syn_std * ref_std + ref_mean

        return x_syn.to(feature_dtype)
