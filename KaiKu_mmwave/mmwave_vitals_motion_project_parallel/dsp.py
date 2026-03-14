import numpy as np


def hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


class RangeFFT:
    def __init__(self, keep_samples: int = 128, fft_bins: int = 64):
        self.keep_samples = int(keep_samples)
        self.fft_bins = int(fft_bins)

    def run(self, raw_frame: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_frame, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"raw_frame must be (RX, CHIRPS, SAMPLES), got {x.shape}")
        _, _, samples = x.shape
        keep = min(self.keep_samples, samples)
        bins = min(self.fft_bins, keep)
        x = x[..., :keep] * hann(keep)[None, None, :]
        return np.fft.fft(x, axis=-1)[..., :bins].astype(np.complex64)


def coherent_bin_vector(F_frame: np.ndarray, bin_idx: int) -> np.ndarray:
    return F_frame[:, :, bin_idx].mean(axis=1)


def combine_rx(rx_vec: np.ndarray, mode: str = "sum") -> complex:
    if mode == "rx0":
        return complex(rx_vec[0])
    if mode == "rx1":
        return complex(rx_vec[1])
    return complex(np.sum(rx_vec))


def detrend_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        return x.copy()
    t = np.arange(x.size, dtype=np.float64)
    p = np.polyfit(t, x, 1)
    return x - (p[0] * t + p[1])


def bandpass_fft(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return x.copy()
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return np.fft.irfft(X * mask, n=x.size)


def dominant_freq(x: np.ndarray, fs: float, f_lo: float, f_hi: float):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return None, 0.0
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return None, 0.0
    mags = np.abs(X)[band]
    fband = freqs[band]
    idx = int(np.argmax(mags))
    return float(fband[idx]), float(mags[idx])


def robust_unwrap_append(last_unwrapped: float, current_angle: float) -> float:
    prev_angle = ((last_unwrapped + np.pi) % (2 * np.pi)) - np.pi
    delta = current_angle - prev_angle
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return float(last_unwrapped + delta)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return 0.6745 * (x - med) / mad
