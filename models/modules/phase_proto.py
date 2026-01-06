import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PhasePrototypeBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_shape: Tuple[int, int, int],
        n_bins: int = 8,
        r_low_ratio: float = 0.2,
        r_high_ratio: float = 0.6,
        use_cos_sin: bool = True,
        ema_beta: float = 0.01,
        amp_source: str = "ref",
        use_amp_stats: bool = False,
        phase_noise_scale: float = 0.0,
        amp_noise_scale: float = 0.0,
        normalize_syn: bool = False,
        replay_detach_ref: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_shape = feat_shape
        self.n_bins = n_bins
        self.r_low_ratio = r_low_ratio
        self.r_high_ratio = r_high_ratio
        self.use_cos_sin = use_cos_sin
        self.ema_beta = ema_beta
        self.amp_source = amp_source
        self.use_amp_stats = use_amp_stats or amp_source == "mean"
        self.phase_noise_scale = phase_noise_scale
        self.amp_noise_scale = amp_noise_scale
        self.normalize_syn = normalize_syn
        self.replay_detach_ref = replay_detach_ref
        self.eps = eps

        c, h, w = feat_shape
        radius_map = self.build_radius_map(h, w, device="cpu")
        mid_mask = self.build_mid_mask(radius_map, r_low_ratio, r_high_ratio)
        ring_edges = self.build_ring_edges(radius_map, r_low_ratio, r_high_ratio, n_bins)
        ring_masks = self.build_ring_masks(radius_map, mid_mask, ring_edges)

        self.register_buffer("mid_mask", mid_mask)
        self.register_buffer("ring_masks", ring_masks)
        self.register_buffer("ring_edges", ring_edges)
        self.register_buffer("phase_cos_bin", torch.zeros(num_classes, c, n_bins))
        self.register_buffer("phase_sin_bin", torch.zeros(num_classes, c, n_bins))
        self.register_buffer("logamp_mean_bin", torch.zeros(num_classes, c, n_bins))
        self.register_buffer("logamp_var_bin", torch.zeros(num_classes, c, n_bins))
        self.register_buffer("count", torch.zeros(num_classes, dtype=torch.long))

    @staticmethod
    def build_radius_map(height: int, width: int, device=None) -> torch.Tensor:
        center_h = (height - 1) / 2.0
        center_w = (width - 1) / 2.0
        y = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
        x = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
        dh = y - center_h
        dw = x - center_w
        r = torch.sqrt(dh ** 2 + dw ** 2)
        return r

    @staticmethod
    def build_mid_mask(
        radius_map: torch.Tensor, r_low_ratio: float, r_high_ratio: float
    ) -> torch.BoolTensor:
        h, w = radius_map.shape
        center_h = (h - 1) / 2.0
        center_w = (w - 1) / 2.0
        r_max = math.sqrt(center_h ** 2 + center_w ** 2)
        r_low = r_low_ratio * r_max
        r_high = r_high_ratio * r_max
        return (r_low <= radius_map) & (radius_map <= r_high)

    @staticmethod
    def build_ring_edges(
        radius_map: torch.Tensor, r_low_ratio: float, r_high_ratio: float, n_bins: int
    ) -> torch.Tensor:
        h, w = radius_map.shape
        center_h = (h - 1) / 2.0
        center_w = (w - 1) / 2.0
        r_max = math.sqrt(center_h ** 2 + center_w ** 2)
        r_low = r_low_ratio * r_max
        r_high = r_high_ratio * r_max
        edges = torch.linspace(r_low, r_high, n_bins + 1, dtype=torch.float32, device=radius_map.device)
        return edges

    @staticmethod
    def build_ring_masks(
        radius_map: torch.Tensor, mid_mask: torch.Tensor, ring_edges: torch.Tensor
    ) -> torch.BoolTensor:
        n_bins = ring_edges.numel() - 1
        h, w = radius_map.shape
        ring_masks = torch.zeros((n_bins, h, w), dtype=torch.bool, device=radius_map.device)
        for b in range(n_bins):
            ring_masks[b] = mid_mask & (radius_map >= ring_edges[b]) & (radius_map < ring_edges[b + 1])
        return ring_masks

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

        if self.n_bins <= 0:
            return False

        with torch.no_grad():
            feature_f = feature.float()
            mask_f = mask.to(feature_f.dtype).unsqueeze(0)
            feat_masked = feature_f * mask_f

            g = torch.fft.fft2(feat_masked)
            g = torch.fft.fftshift(g, dim=(-2, -1))
            amp = torch.abs(g)
            phase = torch.angle(g)

            log_amp = torch.log(amp + self.eps)
            cos_all = torch.cos(phase)
            sin_all = torch.sin(phase)

            cos_bin = torch.zeros((c, self.n_bins), device=feature.device, dtype=feature_f.dtype)
            sin_bin = torch.zeros_like(cos_bin)
            logamp_mean_bin = torch.zeros_like(cos_bin)
            logamp_var_bin = torch.zeros_like(cos_bin)

            for b in range(self.n_bins):
                ring_mask = self.ring_masks[b]
                if ring_mask.sum() == 0:
                    continue
                masked_cos = cos_all[:, ring_mask]
                masked_sin = sin_all[:, ring_mask]
                masked_logamp = log_amp[:, ring_mask]
                cos_bin[:, b] = masked_cos.mean(dim=1)
                sin_bin[:, b] = masked_sin.mean(dim=1)
                logamp_mean_bin[:, b] = masked_logamp.mean(dim=1)
                logamp_var_bin[:, b] = masked_logamp.var(dim=1, unbiased=False)

            if int(self.count[class_id].item()) == 0:
                self.phase_cos_bin[class_id] = cos_bin
                self.phase_sin_bin[class_id] = sin_bin
                self.logamp_mean_bin[class_id] = logamp_mean_bin
                self.logamp_var_bin[class_id] = logamp_var_bin
            else:
                beta = self.ema_beta
                self.phase_cos_bin[class_id] = (1 - beta) * self.phase_cos_bin[class_id] + beta * cos_bin
                self.phase_sin_bin[class_id] = (1 - beta) * self.phase_sin_bin[class_id] + beta * sin_bin
                self.logamp_mean_bin[class_id] = (
                    (1 - beta) * self.logamp_mean_bin[class_id] + beta * logamp_mean_bin
                )
                self.logamp_var_bin[class_id] = (
                    (1 - beta) * self.logamp_var_bin[class_id] + beta * logamp_var_bin
                )

            self.count[class_id] += 1

        return True

    def synthesize_from_reference(
        self,
        feature_ref: torch.Tensor,
        class_id: int,
        use_amp_stats: Optional[bool] = None,
        phase_noise_scale: Optional[float] = None,
        amp_noise_scale: Optional[float] = None,
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
        logamp_ref = torch.log(amp_ref + self.eps)

        phase_syn = phase_ref.clone()
        logamp_syn = logamp_ref.clone()

        use_amp_stats = self.use_amp_stats if use_amp_stats is None else use_amp_stats
        phase_noise_scale = self.phase_noise_scale if phase_noise_scale is None else phase_noise_scale
        amp_noise_scale = self.amp_noise_scale if amp_noise_scale is None else amp_noise_scale

        for b in range(self.n_bins):
            ring_mask = self.ring_masks[b]
            if ring_mask.sum() == 0:
                continue

            ref_cos = torch.cos(phase_ref[:, ring_mask]).mean(dim=1)
            ref_sin = torch.sin(phase_ref[:, ring_mask]).mean(dim=1)
            ref_mean_phase = torch.atan2(ref_sin, ref_cos)

            proto_phase = torch.atan2(self.phase_sin_bin[class_id, :, b], self.phase_cos_bin[class_id, :, b])
            delta = torch.atan2(torch.sin(proto_phase - ref_mean_phase), torch.cos(proto_phase - ref_mean_phase))

            phase_syn[:, ring_mask] = torch.atan2(
                torch.sin(phase_ref[:, ring_mask] + delta[:, None]),
                torch.cos(phase_ref[:, ring_mask] + delta[:, None]),
            )

            if use_amp_stats:
                ref_la_mean = logamp_ref[:, ring_mask].mean(dim=1)
                proto_la_mean = self.logamp_mean_bin[class_id, :, b]
                delta_la = proto_la_mean - ref_la_mean
                logamp_syn[:, ring_mask] = logamp_ref[:, ring_mask] + delta_la[:, None]

            if phase_noise_scale > 0.0:
                cos_mean = self.phase_cos_bin[class_id, :, b]
                sin_mean = self.phase_sin_bin[class_id, :, b]
                r = torch.clamp(torch.sqrt(cos_mean ** 2 + sin_mean ** 2), min=0.0, max=1.0)
                sigma_phase = phase_noise_scale * (1.0 - r)
                eps_phase = torch.randn_like(sigma_phase) * sigma_phase
                phase_syn[:, ring_mask] = torch.atan2(
                    torch.sin(phase_syn[:, ring_mask] + eps_phase[:, None]),
                    torch.cos(phase_syn[:, ring_mask] + eps_phase[:, None]),
                )

            if use_amp_stats and amp_noise_scale > 0.0:
                var_bin = torch.clamp(self.logamp_var_bin[class_id, :, b], min=0.0)
                sigma_la = amp_noise_scale * torch.sqrt(var_bin + self.eps)
                eps_la = torch.randn_like(sigma_la) * sigma_la
                logamp_syn[:, ring_mask] = logamp_syn[:, ring_mask] + eps_la[:, None]

        amp_syn = torch.exp(logamp_syn)

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
