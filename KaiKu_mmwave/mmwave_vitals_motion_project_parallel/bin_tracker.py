import time
import numpy as np


class ResearchBinTracker:
    def __init__(self, bins: int, ignore_bins: int, fps: float, reselect_every_sec: float,
                 hold_sec: float, switch_margin: float, energy_weight: float, phase_snr_weight: float):
        self.bins = bins
        self.ignore_bins = ignore_bins
        self.fps = fps
        self.reselect_every_sec = reselect_every_sec
        self.hold_sec = hold_sec
        self.switch_margin = switch_margin
        self.energy_weight = energy_weight
        self.phase_snr_weight = phase_snr_weight
        self.target_bin = -1
        self.last_pick_t = 0.0
        self.bin_jump = 0.0
        self.prev_energy = None

    def set_fps(self, fps: float):
        self.fps = float(fps)

    def update(self, F: np.ndarray):
        mag = np.abs(F).mean(axis=1)
        energy = mag.sum(axis=0).astype(np.float64)
        if energy.size > self.ignore_bins:
            energy[:self.ignore_bins] = 0.0
        self.energy = energy

        phase_stab = np.zeros_like(energy)
        for b in range(len(energy)):
            z = F[:, :, b].mean(axis=1)
            ang = np.angle(np.sum(z))
            phase_stab[b] = np.abs(np.sum(z)) / (np.mean(np.abs(F[:, :, b])) + 1e-6)
        phase_stab = np.nan_to_num(phase_stab, nan=0.0, posinf=0.0, neginf=0.0)
        self.score = self.energy_weight * energy + self.phase_snr_weight * phase_stab

    def maybe_pick(self, t_now: float = None):
        if t_now is None:
            t_now = time.time()
        if not hasattr(self, "score") or self.score.size == 0:
            return -1, 0.0, []

        idx_sorted = np.argsort(self.score)[::-1]
        top_idx = [int(i) for i in idx_sorted[:5]]
        top_bins = [(int(i), float(self.score[i])) for i in top_idx]
        best = int(idx_sorted[0])
        best_score = float(self.score[best])

        if self.target_bin < 0:
            self.target_bin = best
            self.last_pick_t = t_now
            self.bin_jump = 0.0
            return self.target_bin, best_score, top_bins

        hold_ok = (t_now - self.last_pick_t) >= self.hold_sec
        reselect_ok = (t_now - self.last_pick_t) >= self.reselect_every_sec
        cur_score = float(self.score[self.target_bin]) if self.target_bin < len(self.score) else 0.0
        should_switch = reselect_ok and best != self.target_bin and best_score > (cur_score * self.switch_margin)

        if hold_ok and should_switch:
            self.bin_jump = float(abs(best - self.target_bin))
            self.target_bin = best
            self.last_pick_t = t_now
        else:
            self.bin_jump = 0.0
        return self.target_bin, float(self.score[self.target_bin]), top_bins
