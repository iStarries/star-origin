import torch


def fftshift2d(x: torch.Tensor) -> torch.Tensor:
    """Apply fftshift over the last two dimensions."""
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    """Apply ifftshift over the last two dimensions."""
    return torch.fft.ifftshift(x, dim=(-2, -1))


def get_mid_slices(height: int, width: int, mid_ratio: float):
    mid_h = max(1, int(height * mid_ratio))
    mid_w = max(1, int(width * mid_ratio))
    h0 = (height - mid_h) // 2
    w0 = (width - mid_w) // 2
    h1 = h0 + mid_h
    w1 = w0 + mid_w
    return (slice(h0, h1), slice(w0, w1))


def decompose_spectrum(feature: torch.Tensor):
    spectrum = torch.fft.fft2(feature)
    spectrum = fftshift2d(spectrum)
    amplitude = spectrum.abs()
    phase = torch.angle(spectrum)
    return amplitude, phase


def extract_mid(amplitude: torch.Tensor, phase: torch.Tensor, slices):
    hs, ws = slices
    return amplitude[..., hs, ws], phase[..., hs, ws]


def replace_mid(amplitude: torch.Tensor, phase: torch.Tensor, amp_mid: torch.Tensor, phase_mid: torch.Tensor, slices):
    amplitude_full = amplitude.clone()
    phase_full = phase.clone()
    hs, ws = slices
    amplitude_full[..., hs, ws] = amp_mid
    phase_full[..., hs, ws] = phase_mid
    return amplitude_full, phase_full


def reconstruct_feature(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    spectrum = amplitude * torch.exp(1j * phase)
    spectrum = ifftshift2d(spectrum)
    feature = torch.fft.ifft2(spectrum)
    return feature.real


def mix_phase(phase_ref: torch.Tensor, phase_proto: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return phase_ref
    if alpha >= 1:
        return phase_proto
    return (1 - alpha) * phase_ref + alpha * phase_proto
