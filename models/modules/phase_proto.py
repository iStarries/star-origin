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
        use_amp_stats: bool = True,
        phase_noise_scale: float = 0.0,
        amp_noise_scale: float = 0.0,
        normalize_syn: bool = False,
        replay_detach_ref: bool = True,
        eps: float = 1e-6,
        phase_model: str = "gmm",
        phase_gmm_components: int = 3,
        phase_gmm_diag: bool = True,
        phase_gmm_min_var: float = 1e-4,
        phase_residual_scale: float = 0.5,
        phase_sample_mode: str = "mixture",
        phase_update_mode: str = "ema_em",
        phase_use_delta: bool = True,
        **_: object,
    ):
        super().__init__()
        if phase_model != "gmm":
            raise ValueError(f"[PHASE-PPB] Unsupported phase_model={phase_model}. Only 'gmm' is supported.")
        if not phase_gmm_diag:
            raise ValueError("[PHASE-PPB] Only diagonal phase GMM covariance is supported.")
        if phase_gmm_components <= 0:
            raise ValueError("[PHASE-PPB] phase_gmm_components must be > 0.")
        if phase_sample_mode not in {"mixture", "argmax"}:
            raise ValueError(
                f"[PHASE-PPB] Unsupported phase_sample_mode={phase_sample_mode}. "
                "Use 'mixture' or 'argmax'."
            )
        if phase_update_mode != "ema_em":
            raise ValueError(
                f"[PHASE-PPB] Unsupported phase_update_mode={phase_update_mode}. Only 'ema_em' is supported."
            )

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
        self.phase_model = phase_model
        self.phase_gmm_components = int(phase_gmm_components)
        self.phase_gmm_diag = phase_gmm_diag
        self.phase_gmm_min_var = float(phase_gmm_min_var)
        self.phase_residual_scale = float(phase_residual_scale)
        self.phase_sample_mode = phase_sample_mode
        self.phase_update_mode = phase_update_mode
        self.phase_use_delta = phase_use_delta

        c, h, w = feat_shape
        radius_map = self.build_radius_map(h, w, device="cpu")
        mid_mask = self.build_mid_mask(radius_map, r_low_ratio, r_high_ratio)
        ring_edges = self.build_ring_edges(radius_map, r_low_ratio, r_high_ratio, n_bins)
        ring_masks = self.build_ring_masks(radius_map, mid_mask, ring_edges)

        self.register_buffer("mid_mask", mid_mask)
        self.register_buffer("ring_masks", ring_masks)
        self.register_buffer("ring_edges", ring_edges)
        self.register_buffer("logamp_mean_bin", torch.zeros(num_classes, c, n_bins))
        self.register_buffer("logamp_var_bin", torch.zeros(num_classes, c, n_bins))
        self.register_buffer("count", torch.zeros(num_classes, dtype=torch.long))
        # phase_gmm_* stores the class/channel GMM in cos/sin space.
        self.register_buffer("phase_gmm_weight", torch.zeros(num_classes, c, self.phase_gmm_components))
        self.register_buffer("phase_gmm_mean", torch.zeros(num_classes, c, self.phase_gmm_components, 2))
        self.register_buffer(
            "phase_gmm_var",
            torch.full((num_classes, c, self.phase_gmm_components, 2), self.phase_gmm_min_var),
        )
        self.register_buffer("phase_gmm_count", torch.zeros(num_classes, c, self.phase_gmm_components))

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

    def _normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(self.eps)
        return vectors / norm

    def _class_has_any_phase_stats(self, class_id: int) -> bool:
        return bool((self.phase_gmm_count[class_id].sum() > 0).item())

    def _class_has_valid_phase_gmm(self, class_id: int) -> bool:
        counts = self.phase_gmm_count[class_id]
        if counts.numel() == 0 or not bool((counts.sum(dim=1) > 0).all().item()):
            return False

        weight = self.phase_gmm_weight[class_id]
        mean = self.phase_gmm_mean[class_id]
        var = self.phase_gmm_var[class_id]
        if not torch.isfinite(weight).all() or not torch.isfinite(mean).all() or not torch.isfinite(var).all():
            return False
        if bool((var < self.phase_gmm_min_var).any().item()):
            return False

        weight_sum = weight.sum(dim=-1)
        if bool((weight_sum <= 0).any().item()):
            return False
        return True

    def _legacy_amp_only_ready(self, class_id: int) -> bool:
        return int(self.count[class_id].item()) > 0 and not self._class_has_any_phase_stats(class_id)

    def is_ready(self, class_id: int) -> bool:
        if class_id < 0 or class_id >= self.num_classes:
            raise IndexError(
                f"[PHASE-PPB] class_id={class_id} out of range for num_classes={self.num_classes}. "
                f"This indicates mismatch between train_id mapping/head channels and PPB num_classes."
            )
        if int(self.count[class_id].item()) <= 0:
            return False
        return self._legacy_amp_only_ready(class_id) or self._class_has_valid_phase_gmm(class_id)

    def get_mid_size(self) -> int:
        return int(self.mid_mask.sum().item())

    def _init_phase_gmm_from_samples(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c, n_mid, _ = samples.shape
        if n_mid <= 0:
            raise ValueError("[PHASE-PPB] Cannot initialize phase GMM with empty mid-band samples.")

        indices = torch.linspace(
            0,
            max(n_mid - 1, 0),
            steps=self.phase_gmm_components,
            device=samples.device,
        ).round().long()
        mean = samples[:, indices, :].clone()
        if n_mid > 1:
            base_var = samples.var(dim=1, unbiased=False, keepdim=True)
        else:
            base_var = torch.full((c, 1, 2), self.phase_gmm_min_var, dtype=samples.dtype, device=samples.device)
        var = torch.clamp(base_var.expand(-1, self.phase_gmm_components, -1).clone(), min=self.phase_gmm_min_var)
        weight = torch.full(
            (c, self.phase_gmm_components),
            1.0 / float(self.phase_gmm_components),
            dtype=samples.dtype,
            device=samples.device,
        )
        return weight, mean, var

    def _compute_phase_batch_stats(
        self,
        samples: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.clamp(weight, min=self.eps)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        var = torch.clamp(var, min=self.phase_gmm_min_var)

        diff = samples.unsqueeze(2) - mean.unsqueeze(1)
        log_weight = torch.log(weight.unsqueeze(1).clamp_min(self.eps))
        logits = log_weight - 0.5 * (torch.log(var.unsqueeze(1)) + diff.pow(2) / var.unsqueeze(1)).sum(dim=-1)
        # EMA-EM update: use current GMM to compute responsibilities, then update batch stats.
        resp = torch.softmax(logits, dim=-1)
        eff_count = resp.sum(dim=1)
        valid = eff_count > self.eps

        batch_weight = eff_count / float(samples.shape[1])
        batch_mean = torch.einsum("cnk,cnm->ckm", resp, samples)
        batch_mean = batch_mean / eff_count.unsqueeze(-1).clamp_min(self.eps)
        centered = samples.unsqueeze(2) - batch_mean.unsqueeze(1)
        batch_var = torch.einsum("cnk,cnkm->ckm", resp, centered.pow(2))
        batch_var = batch_var / eff_count.unsqueeze(-1).clamp_min(self.eps)
        batch_var = torch.clamp(batch_var, min=self.phase_gmm_min_var)
        return valid, batch_weight, batch_mean, batch_var

    def _update_phase_gmm(self, class_id: int, samples: torch.Tensor) -> None:
        has_phase = self._class_has_any_phase_stats(class_id)
        if has_phase:
            prev_weight = self.phase_gmm_weight[class_id]
            prev_mean = self.phase_gmm_mean[class_id]
            prev_var = self.phase_gmm_var[class_id]
        else:
            prev_weight, prev_mean, prev_var = self._init_phase_gmm_from_samples(samples)

        valid, batch_weight, batch_mean, batch_var = self._compute_phase_batch_stats(
            samples=samples,
            weight=prev_weight,
            mean=prev_mean,
            var=prev_var,
        )

        beta = self.ema_beta if has_phase else 1.0
        valid_expanded = valid.unsqueeze(-1)
        next_weight = torch.where(valid, (1 - beta) * prev_weight + beta * batch_weight, prev_weight)
        next_mean = torch.where(valid_expanded, (1 - beta) * prev_mean + beta * batch_mean, prev_mean)
        next_var = torch.where(valid_expanded, (1 - beta) * prev_var + beta * batch_var, prev_var)
        next_var = torch.clamp(next_var, min=self.phase_gmm_min_var)
        next_weight = next_weight / next_weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        self.phase_gmm_weight[class_id].copy_(next_weight)
        self.phase_gmm_mean[class_id].copy_(next_mean)
        self.phase_gmm_var[class_id].copy_(next_var)
        self.phase_gmm_count[class_id].add_(
            torch.where(valid, torch.clamp(batch_weight * samples.shape[1], min=0.0), torch.zeros_like(batch_weight))
        )

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
        if (c, h, w) != self.feat_shape or mask.ndim != 2 or mask.shape != (h, w):
            return False
        if mask.sum() == 0 or self.n_bins <= 0:
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
            logamp_mean_bin = torch.zeros((c, self.n_bins), device=feature.device, dtype=feature_f.dtype)
            logamp_var_bin = torch.zeros_like(logamp_mean_bin)
            for b in range(self.n_bins):
                ring_mask = self.ring_masks[b]
                if not bool(ring_mask.any().item()):
                    continue
                masked_logamp = log_amp[:, ring_mask]
                logamp_mean_bin[:, b] = masked_logamp.mean(dim=1)
                logamp_var_bin[:, b] = masked_logamp.var(dim=1, unbiased=False)

            if int(self.count[class_id].item()) == 0:
                self.logamp_mean_bin[class_id] = logamp_mean_bin
                self.logamp_var_bin[class_id] = logamp_var_bin
            else:
                beta = self.ema_beta
                self.logamp_mean_bin[class_id] = (
                    (1 - beta) * self.logamp_mean_bin[class_id] + beta * logamp_mean_bin
                )
                self.logamp_var_bin[class_id] = (
                    (1 - beta) * self.logamp_var_bin[class_id] + beta * logamp_var_bin
                )

            phase_samples = torch.stack((torch.cos(phase[:, self.mid_mask]), torch.sin(phase[:, self.mid_mask])), dim=-1)
            phase_samples = self._normalize_vectors(phase_samples)
            self._update_phase_gmm(class_id, phase_samples)
            self.count[class_id] += 1

        return True

    def _sample_phase_targets(self, class_id: int, n_mid: int, device: torch.device) -> Optional[torch.Tensor]:
        weight = self.phase_gmm_weight[class_id]
        mean = self.phase_gmm_mean[class_id]
        var = torch.clamp(self.phase_gmm_var[class_id], min=self.phase_gmm_min_var)

        if not torch.isfinite(weight).all() or not torch.isfinite(mean).all() or not torch.isfinite(var).all():
            return None
        if bool((weight < 0).any().item()):
            return None

        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if self.phase_sample_mode == "argmax":
            comp_idx = weight.argmax(dim=-1, keepdim=True).expand(-1, n_mid)
        else:
            cumulative = weight.cumsum(dim=-1).unsqueeze(1)
            rand = torch.rand(weight.shape[0], n_mid, 1, device=device)
            comp_idx = (rand > cumulative).sum(dim=-1).clamp(max=self.phase_gmm_components - 1)

        gather_index = comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        chosen_mean = torch.gather(mean.unsqueeze(1).expand(-1, n_mid, -1, -1), 2, gather_index).squeeze(2)
        chosen_var = torch.gather(var.unsqueeze(1).expand(-1, n_mid, -1, -1), 2, gather_index).squeeze(2)
        sampled = chosen_mean + torch.randn_like(chosen_mean) * torch.sqrt(chosen_var)
        sampled = self._normalize_vectors(sampled)
        return sampled if torch.isfinite(sampled).all() else None

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
        if (c, h, w) != self.feat_shape or not self.is_ready(class_id):
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

        phase_noise_scale = self.phase_noise_scale if phase_noise_scale is None else phase_noise_scale
        amp_noise_scale = self.amp_noise_scale if amp_noise_scale is None else amp_noise_scale

        has_phase_stats = self._class_has_any_phase_stats(class_id)
        has_valid_phase_gmm = self._class_has_valid_phase_gmm(class_id)
        legacy_amp_only = int(self.count[class_id].item()) > 0 and not has_phase_stats
        if has_phase_stats and not has_valid_phase_gmm:
            return None

        # Legacy checkpoints have no phase GMM, so replay falls back to amplitude-only perturbation.
        apply_amp_stats = self.use_amp_stats if use_amp_stats is None else use_amp_stats
        if legacy_amp_only:
            apply_amp_stats = True

        for b in range(self.n_bins):
            ring_mask = self.ring_masks[b]
            if not bool(ring_mask.any().item()):
                continue
            if apply_amp_stats:
                ref_la_mean = logamp_ref[:, ring_mask].mean(dim=1)
                proto_la_mean = self.logamp_mean_bin[class_id, :, b]
                delta_la = proto_la_mean - ref_la_mean
                logamp_syn[:, ring_mask] = logamp_ref[:, ring_mask] + delta_la[:, None]
                if amp_noise_scale > 0.0:
                    var_bin = torch.clamp(self.logamp_var_bin[class_id, :, b], min=0.0)
                    sigma_la = amp_noise_scale * torch.sqrt(var_bin + self.eps)
                    logamp_syn[:, ring_mask] = logamp_syn[:, ring_mask] + torch.randn_like(sigma_la)[:, None] * sigma_la[:, None]

        if has_valid_phase_gmm:
            ref_vectors = torch.stack(
                (torch.cos(phase_ref[:, self.mid_mask]), torch.sin(phase_ref[:, self.mid_mask])),
                dim=-1,
            )
            ref_vectors = self._normalize_vectors(ref_vectors)
            target_vectors = self._sample_phase_targets(class_id, ref_vectors.shape[1], feature_ref.device)
            if target_vectors is None:
                return None

            if self.phase_use_delta:
                delta_vectors = target_vectors - ref_vectors
            else:
                delta_vectors = target_vectors

            # Replay keeps the reference phase as the skeleton and only injects residual cos/sin offsets.
            syn_vectors = ref_vectors + self.phase_residual_scale * delta_vectors
            if phase_noise_scale > 0.0:
                syn_vectors = syn_vectors + torch.randn_like(syn_vectors) * phase_noise_scale
            syn_vectors = self._normalize_vectors(syn_vectors)
            if not torch.isfinite(syn_vectors).all():
                return None
            phase_syn[:, self.mid_mask] = torch.atan2(syn_vectors[..., 1], syn_vectors[..., 0])

        amp_syn = torch.exp(logamp_syn)
        g_syn = torch.polar(amp_syn, phase_syn)
        g_syn = torch.fft.ifftshift(g_syn, dim=(-2, -1))
        x_syn = torch.fft.ifft2(g_syn).real
        if not torch.isfinite(x_syn).all():
            return None

        if self.normalize_syn:
            ref_mean = feature_ref.mean(dim=(1, 2), keepdim=True)
            ref_std = feature_ref.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
            syn_mean = x_syn.mean(dim=(1, 2), keepdim=True)
            syn_std = x_syn.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
            x_syn = (x_syn - syn_mean) / syn_std * ref_std + ref_mean

        return x_syn.to(feature_dtype)
