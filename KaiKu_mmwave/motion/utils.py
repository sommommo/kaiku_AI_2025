from collections import deque
import numpy as np


def to_int_safe(x, default=-1):
    try:
        if hasattr(x, "data"):
            x = x.data
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


class RingArray:
    def __init__(self, maxlen: int):
        self.data = deque(maxlen=maxlen)

    def append(self, x):
        self.data.append(x)

    def as_array(self):
        if len(self.data) == 0:
            return np.array([])
        first = np.asarray(self.data[0])
        if first.ndim == 0:
            return np.asarray(list(self.data))
        return np.stack(list(self.data), axis=0)

    def __len__(self):
        return len(self.data)


class SignalTools:
    @staticmethod
    def detrend(x):
        x = np.asarray(x, dtype=np.float64)
        if x.size < 4:
            return x
        t = np.arange(x.size, dtype=np.float64)
        p = np.polyfit(t, x, 1)
        return x - (p[0] * t + p[1])

    @staticmethod
    def bandpass_fft(x, fs, f_lo, f_hi):
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        if n < 8:
            return x.copy()
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        return np.fft.irfft(X * mask, n=n)

    @staticmethod
    def dominant_freq(x, fs, f_lo, f_hi):
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        if n < 16:
            return None, 0.0
        w = np.hanning(n)
        X = np.fft.rfft(x * w)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(band):
            return None, 0.0
        mags = np.abs(X)[band]
        fband = freqs[band]
        i = int(np.argmax(mags))
        return float(fband[i]), float(mags[i])

    @staticmethod
    def robust_z(x):
        x = np.asarray(x, dtype=np.float64)
        if x.size < 8:
            return np.zeros_like(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        return 0.6745 * (x - med) / mad

    @staticmethod
    def count_bursts(x, threshold):
        x = np.asarray(x, dtype=np.float64)
        flag = x > threshold
        if x.size == 0:
            return 0
        return int(np.sum((~flag[:-1]) & flag[1:])) if x.size > 1 else int(flag[0])
