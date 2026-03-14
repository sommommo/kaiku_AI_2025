from collections import deque
import numpy as np
from dsp import detrend_linear, bandpass_fft, dominant_freq


class ResearchVitalsEstimator:
    def __init__(self, fs: float, window_sec: float, estimate_min_ratio: float, resp_band, heart_band):
        self.fs = fs
        self.window_sec = window_sec
        self.estimate_min_ratio = estimate_min_ratio
        self.resp_band = resp_band
        self.heart_band = heart_band
        self.N = max(16, int(window_sec * fs))
        self.phase_buf = deque(maxlen=self.N)

    def set_fps(self, fs: float, window_sec: float):
        self.fs = fs
        self.window_sec = window_sec
        self.N = max(16, int(window_sec * fs))
        self.phase_buf = deque(self.phase_buf, maxlen=self.N)

    def update(self, phase_unwrapped: float):
        self.phase_buf.append(float(phase_unwrapped))

    def estimate(self):
        x = np.asarray(self.phase_buf, dtype=np.float64)
        need_len = int(self.estimate_min_ratio * self.N)
        dbg = {"buf_len": int(x.size), "need_len": int(need_len)}
        if x.size < max(16, need_len):
            return None, None, np.array([]), np.array([]), dbg

        x = detrend_linear(x)
        resp = bandpass_fft(x, self.fs, *self.resp_band)
        heart = bandpass_fft(x, self.fs, *self.heart_band)

        fr, mr = dominant_freq(resp, self.fs, *self.resp_band)
        fh, mh = dominant_freq(heart, self.fs, *self.heart_band)
        rpm = None if fr is None else fr * 60.0
        bpm = None if fh is None else fh * 60.0
        dbg.update({"resp_mag": mr, "heart_mag": mh})
        return rpm, bpm, resp, heart, dbg
