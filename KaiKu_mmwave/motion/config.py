from dataclasses import dataclass
from typing import Tuple
import os
import time


@dataclass
class SystemConfig:
    setting_dir: str = r"C:\Users\USER\Desktop\KaiKu_mmwave\radar-gesture-recognition-chore-update-20250815\TempParam\K60168-Test-00256-008-v0.0.8-20230717_120cm"
    stream_type: str = "raw_data"
    out_root: str = "OutputResearchSystem"
    dataset_root: str = os.path.join("OutputResearchSystem", "dataset_master")

    frame_buffer_max: int = 2400
    fft_bins: int = 64
    ignore_bins: int = 3
    topk_bins: int = 5
    auto_bin_smooth: float = 0.85
    auto_reselect_sec: float = 0.4

    fps_est: float = 28.0
    phase_window_sec: float = 20.0
    event_pre_sec: float = 1.5
    event_post_sec: float = 2.5

    resp_band: Tuple[float, float] = (0.10, 0.60)
    heart_band: Tuple[float, float] = (0.80, 3.00)
    cough_band: Tuple[float, float] = (0.80, 4.50)

    motion_z_fall: float = 4.5
    motion_z_cough: float = 2.8
    cough_min_bursts: int = 2
    fall_still_sec: float = 2.0
    low_motion_z: float = 0.8

    enable_dataset_recorder: bool = True
    max_gui_points: int = 800


class SessionPaths:
    def __init__(self, cfg: SystemConfig):
        self.session_name = time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(cfg.out_root, self.session_name)
        self.raw_save_path = os.path.join(self.session_dir, "raw_frames.npz")
        self.meta_save_path = os.path.join(self.session_dir, "session_meta.json")

        self.dataset_root = cfg.dataset_root
        self.dataset_index_path = os.path.join(self.dataset_root, "index.csv")
        self.dataset_label_root = os.path.join(self.dataset_root, "segments")

        # backward compatibility
        self.dataset_dir = self.dataset_label_root

    def ensure(self):
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.dataset_root, exist_ok=True)
        os.makedirs(self.dataset_label_root, exist_ok=True)


KEY_LABELS = {
    "1": "fall",
    "2": "cough",
    "3": "sit",
    "4": "normal",
    "5": "bend",
    "6": "walk",
}
