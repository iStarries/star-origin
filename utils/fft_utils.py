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


def _enforce_hermitian_unshifted(spectrum: torch.Tensor) -> torch.Tensor:
    """Enforce Hermitian symmetry on an unshifted 2D spectrum.

    For real-valued spatial signals, the Fourier spectrum should satisfy:
        S[k] = conj(S[-k])
    in each frequency dimension (with wrapping).

    This function symmetrizes the complex spectrum to make the inverse FFT
    produce (approximately) real-valued outputs without discarding the
    imaginary part ad-hoc.

    Args:
        spectrum: Complex tensor of shape [..., H, W] in unshifted FFT layout.

    Returns:
        Complex tensor with enforced Hermitian symmetry.
    """
    if not torch.is_complex(spectrum):
        raise TypeError("spectrum must be a complex tensor")

    H, W = spectrum.shape[-2], spectrum.shape[-1]
    # Mirror index mapping: (i, j) -> (-i mod H, -j mod W)
    mirror = spectrum.flip(-2).flip(-1)
    mirror = torch.roll(mirror, shifts=1, dims=-2)
    mirror = torch.roll(mirror, shifts=1, dims=-1)
    spectrum = 0.5 * (spectrum + mirror.conj())

    # Force special frequency bins to be purely real.
    def _force_real(i: int, j: int):
        v = spectrum[..., i, j].real
        spectrum[..., i, j] = torch.complex(v, torch.zeros_like(v))

    _force_real(0, 0)
    if H % 2 == 0:
        _force_real(H // 2, 0)
    if W % 2 == 0:
        _force_real(0, W // 2)
    if H % 2 == 0 and W % 2 == 0:
        _force_real(H // 2, W // 2)

    return spectrum


def reconstruct_feature(amplitude: torch.Tensor, phase: torch.Tensor, enforce_hermitian: bool = False) -> torch.Tensor:
    """Reconstruct spatial features from amplitude/phase.

    Args:
        amplitude: Real tensor in shifted FFT layout.
        phase: Real tensor in shifted FFT layout.
        enforce_hermitian: If True, enforce Hermitian symmetry on the
            unshifted spectrum before ifft2 (recommended when the spectrum is
            constructed/edited manually, e.g., phase replay).

    Returns:
        Real tensor (spatial feature map).
    """
    spectrum = amplitude * torch.exp(1j * phase)
    spectrum = ifftshift2d(spectrum)
    if enforce_hermitian:
        spectrum = _enforce_hermitian_unshifted(spectrum)
    feature = torch.fft.ifft2(spectrum)
    return feature.real


def mix_phase(phase_ref: torch.Tensor, phase_proto: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return phase_ref
    if alpha >= 1:
        return phase_proto
    return (1 - alpha) * phase_ref + alpha * phase_proto
