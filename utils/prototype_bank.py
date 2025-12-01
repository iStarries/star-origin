from typing import Dict, Optional

import torch

from utils.fft_utils import decompose_spectrum, extract_mid, get_mid_slices


class PhasePrototypeBank:
    def __init__(self, mid_ratio: float = 0.5, momentum: float = 0.01, device: Optional[torch.device] = None):
        self.mid_ratio = mid_ratio
        self.momentum = momentum
        self.device = device
        self._store: Dict[int, Dict[str, torch.Tensor]] = {}
        self._mid_slices = None

    def _init_entry(self, class_id: int, mid_shape):
        # mid_shape may include a batch dimension when called with batched features.
        c, h, w = list(mid_shape)[-3:]
        self._store[class_id] = {
            'cos': torch.zeros((c, h, w), device=self.device),
            'sin': torch.zeros((c, h, w), device=self.device),
            'amp_mean': torch.zeros((c, h, w), device=self.device),
            'amp_M2': torch.zeros((c, h, w), device=self.device),
            'count': torch.tensor(0, device=self.device, dtype=torch.long),
        }

    def _get_mid_slices(self, feature: torch.Tensor):
        if self._mid_slices is None:
            _, _, h, w = feature.shape
            self._mid_slices = get_mid_slices(h, w, self.mid_ratio)
        return self._mid_slices

    def update_from_masked_feature(self, masked_feature: torch.Tensor, class_id: int):
        if masked_feature.numel() == 0:
            return
        mid_slices = self._get_mid_slices(masked_feature)
        amplitude, phase = decompose_spectrum(masked_feature)
        amp_mid, phase_mid = extract_mid(amplitude, phase, mid_slices)

        # Collapse any leading dimensions so the prototype keeps shape (C, H, W).
        while amp_mid.dim() > 3:
            amp_mid = amp_mid.mean(dim=0)
            phase_mid = phase_mid.mean(dim=0)

        if class_id not in self._store:
            self._init_entry(class_id, amp_mid.shape)

        entry = self._store[class_id]
        beta = self.momentum
        cos_new = torch.cos(phase_mid).detach()
        sin_new = torch.sin(phase_mid).detach()
        entry['cos'].mul_(1 - beta).add_(beta * cos_new)
        entry['sin'].mul_(1 - beta).add_(beta * sin_new)

        amp_new = amp_mid.detach()
        entry['count'] += 1
        delta = amp_new - entry['amp_mean']
        entry['amp_mean'] += delta / entry['count'].clamp_min(1)
        delta2 = amp_new - entry['amp_mean']
        entry['amp_M2'] += delta * delta2

    def has_class(self, class_id: int) -> bool:
        return class_id in self._store

    def get_phase(self, class_id: int):
        entry = self._store[class_id]
        return torch.atan2(entry['sin'], entry['cos'])

    def get_amp_stats(self, class_id: int):
        entry = self._store[class_id]
        count = entry['count'].clamp_min(1).float()
        var = entry['amp_M2'] / count
        std = torch.sqrt(var + 1e-6)
        return entry['amp_mean'], std

    def state_dict(self):
        return {
            'mid_ratio': self.mid_ratio,
            'momentum': self.momentum,
            'store': self._store,
            'mid_slices': self._mid_slices,
        }

    def load_state_dict(self, state_dict, map_location=None):
        self.mid_ratio = state_dict.get('mid_ratio', self.mid_ratio)
        self.momentum = state_dict.get('momentum', self.momentum)
        self._mid_slices = state_dict.get('mid_slices', None)
        store = state_dict.get('store', {})
        if map_location is not None:
            self._store = {
                k: {kk: vv.to(map_location) for kk, vv in v.items()} for k, v in store.items()
            }
        else:
            self._store = store

    def to(self, device):
        self.device = device
        for entry in self._store.values():
            for key, tensor in entry.items():
                entry[key] = tensor.to(device)
        return self
