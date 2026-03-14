from collections import deque
import time
import numpy as np
from dsp import robust_z


class MotionSuppressor:
    def __init__(self, fs: float, short_sec: float, phase_std_thr: float, bin_jump_thr: float, freeze_sec: float):
        self.fs = fs
        self.short_sec = short_sec
        self.phase_std_thr = phase_std_thr
        self.bin_jump_thr = bin_jump_thr
        self.freeze_sec = freeze_sec
        self.phase_buf = deque(maxlen=max(8, int(short_sec * fs)))
        self.motion_buf = deque(maxlen=max(32, int(4 * fs)))
        self.last_motion_t = 0.0

    def set_fps(self, fs: float, short_sec: float, freeze_sec: float):
        self.fs = fs
        self.short_sec = short_sec
        self.freeze_sec = freeze_sec
        self.phase_buf = deque(self.phase_buf, maxlen=max(8, int(short_sec * fs)))
        self.motion_buf = deque(self.motion_buf, maxlen=max(32, int(4 * fs)))

    def update(self, phase_unwrapped: float, bin_jump: float):
        now = time.time()
        self.phase_buf.append(float(phase_unwrapped))
        phase_std = float(np.std(self.phase_buf)) if len(self.phase_buf) >= 4 else 0.0
        motion_level = phase_std + 0.5 * float(bin_jump)
        self.motion_buf.append(motion_level)
        mz = robust_z(np.asarray(self.motion_buf, dtype=np.float64)) if len(self.motion_buf) >= 8 else np.array([0.0])
        motion_z = float(mz[-1]) if mz.size else 0.0
        moving = phase_std > self.phase_std_thr or bin_jump >= self.bin_jump_thr or motion_z > 3.0
        if moving:
            self.last_motion_t = now
        motion_flag = (now - self.last_motion_t) < self.freeze_sec
        stable_sec = max(0.0, now - self.last_motion_t)
        return motion_level, motion_flag, stable_sec, phase_std, motion_z
