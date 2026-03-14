from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from config import SystemConfig
from utils import SignalTools


@dataclass
class Summary:
    fps: float
    target_bin: Optional[int]
    top_bins: List[Tuple[int, float]]
    range_score: np.ndarray
    motion: float
    motion_z: float
    phase_series: np.ndarray
    resp_signal: np.ndarray
    heart_signal: np.ndarray
    rpm: Optional[float]
    bpm: Optional[float]
    phase_std: float
    cough_bursts: int


class FeatureExtractor:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.range_score_smooth = None
        self.target_bin = None
        self.last_bin_update_t = 0.0
        self.last_raw = None
        self.rx_count = None
        self.chirp_count = None
        self.sample_count = None

    def set_shape_if_needed(self, raw: np.ndarray):
        if self.rx_count is None:
            self.rx_count, self.chirp_count, self.sample_count = raw.shape

    def compute_motion_energy(self, raw: np.ndarray) -> float:
        x = raw.astype(np.float32)
        if self.last_raw is None:
            self.last_raw = x.copy()
            return 0.0
        e = float(np.mean(np.abs(x - self.last_raw)))
        self.last_raw = x.copy()
        return e

    def range_fft(self, raw: np.ndarray) -> np.ndarray:
        keep = min(self.sample_count or raw.shape[-1], raw.shape[-1])
        bins = min(self.cfg.fft_bins, keep)
        return np.fft.fft(raw[..., :keep].astype(np.float32), axis=-1)[..., :bins]

    def range_profile_score(self, fft_frame: np.ndarray) -> np.ndarray:
        mag = np.abs(fft_frame).mean(axis=1)
        score = mag.sum(axis=0)
        score = score.astype(np.float64)
        if score.size > self.cfg.ignore_bins:
            score[:self.cfg.ignore_bins] = 0.0
        return score

    def pick_target_bin(self, score: np.ndarray, now: float) -> Optional[int]:
        if score.size == 0:
            return None
        if self.range_score_smooth is None:
            self.range_score_smooth = score.copy()
        else:
            a = self.cfg.auto_bin_smooth
            self.range_score_smooth = a * self.range_score_smooth + (1.0 - a) * score

        if (now - self.last_bin_update_t) < self.cfg.auto_reselect_sec and self.target_bin is not None:
            return self.target_bin

        best = int(np.argmax(self.range_score_smooth))
        self.target_bin = best
        self.last_bin_update_t = now
        return best

    def top_bins(self, score: np.ndarray) -> List[Tuple[int, float]]:
        s = score.copy()
        if s.size == 0:
            return []
        s[:self.cfg.ignore_bins] = -1
        idx = np.argsort(s)[::-1][:self.cfg.topk_bins]
        return [(int(i), float(score[i])) for i in idx]

    def phase_from_fft_frame(self, fft_frame: np.ndarray, bin_idx: int) -> float:
        s = fft_frame[:, :, bin_idx].mean(axis=1)
        z = np.sum(s)
        return float(np.angle(z))

    def phase_series_from_history(self, phase_hist: np.ndarray) -> np.ndarray:
        ph = np.asarray(phase_hist, dtype=np.float64)
        if ph.size == 0:
            return ph
        valid = np.isfinite(ph)
        if np.sum(valid) < 4:
            return np.nan_to_num(ph)
        idx = np.arange(ph.size)
        ph[~valid] = np.interp(idx[~valid], idx[valid], ph[valid])
        ph = np.unwrap(ph)
        return SignalTools.detrend(ph)

    def build_summary(self, fps: float, phase_hist: np.ndarray, motion_hist: np.ndarray, score: np.ndarray, motion: float) -> Summary:
        phase_series = self.phase_series_from_history(phase_hist)
        resp_sig, heart_sig = np.array([]), np.array([])
        rpm, bpm = None, None

        if phase_series.size >= max(32, int(4 * fps)):
            ph_win = phase_series[-min(len(phase_series), int(self.cfg.phase_window_sec * fps)):]
            resp_sig = SignalTools.bandpass_fft(ph_win, fps, *self.cfg.resp_band)
            heart_sig = SignalTools.bandpass_fft(ph_win, fps, *self.cfg.heart_band)
            f_r, _ = SignalTools.dominant_freq(resp_sig, fps, *self.cfg.resp_band)
            f_h, _ = SignalTools.dominant_freq(heart_sig, fps, *self.cfg.heart_band)
            rpm = None if f_r is None else f_r * 60.0
            bpm = None if f_h is None else f_h * 60.0

        motion_z = 0.0
        cough_bursts = 0
        motion_hist = np.asarray(motion_hist, dtype=np.float64)
        if motion_hist.size >= 16:
            mz = SignalTools.robust_z(motion_hist)
            motion_z = float(mz[-1])
            short = motion_hist[-min(len(motion_hist), int(2.0 * fps)):]
            cough_bursts = SignalTools.count_bursts(SignalTools.robust_z(short), self.cfg.motion_z_cough)

        return Summary(
            fps=fps,
            target_bin=self.target_bin,
            top_bins=self.top_bins(score),
            range_score=score,
            motion=float(motion),
            motion_z=motion_z,
            phase_series=phase_series,
            resp_signal=resp_sig,
            heart_signal=heart_sig,
            rpm=rpm,
            bpm=bpm,
            phase_std=float(np.std(phase_series[-min(phase_series.size, int(4 * fps)):])) if phase_series.size > 0 else 0.0,
            cough_bursts=cough_bursts,
        )
