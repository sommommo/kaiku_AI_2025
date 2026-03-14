from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
import os


@dataclass
class DeviceConfig:
    setting_dir: str = r"C:\Users\113\Desktop\KaiKu_mmwave\radar-gesture-recognition-chore-update-20250815\TempParam\K60168-Test-00256-008-v0.0.8-20230717_120cm"
    stream_type: str = "raw_data"
    use_kkt_device: bool = True


@dataclass
class DSPConfig:
    keep_samples: int = 128
    fft_bins: int = 64
    rx_mode: str = "sum"              # sum / rx0 / rx1
    ignore_bins: int = 6
    topk_bins: int = 5
    fps_init: float = 28.0

    window_sec: float = 20.0
    estimate_min_ratio: float = 0.65

    reselect_every_sec: float = 0.5
    bin_hold_sec: float = 1.8
    switch_margin: float = 1.12
    energy_weight: float = 0.70
    phase_snr_weight: float = 0.30

    motion_short_sec: float = 1.2
    motion_thr_phase_std: float = 2.0
    motion_thr_bin_jump: float = 2.0
    motion_freeze_sec: float = 1.5

    resp_band: Tuple[float, float] = (0.10, 0.60)
    heart_band: Tuple[float, float] = (0.80, 3.00)
    motion_band: Tuple[float, float] = (0.60, 5.00)


@dataclass
class ModelConfig:
    enabled: bool = True
    model_path: Optional[str] = None
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "normal",
        1: "fall",
        2: "cough",
        3: "sit",
        4: "bend",
        5: "walk",
        6: "other",
    })
    infer_window_sec: float = 2.0
    infer_stride_sec: float = 0.25
    score_threshold: float = 0.50
    motion_gate_threshold: float = 3.0
    idle_label: str = "normal" ## "normal" 會顯示與名稱相同的圖片
    

@dataclass
class GUIConfig:
    title: str = "mmWave Vitals + Abnormal Motion Detection"
    refresh_ms: int = 80
    max_points: int = 800
    action_asset_dir: str = os.path.join("assets", "actions")
    alert_asset_dir: str = os.path.join("assets", "alerts")
    resp_normal_range: Tuple[float, float] = (8.0, 24.0)
    heart_normal_range: Tuple[float, float] = (50.0, 120.0)
    main_width: int = 1480
    main_height: int = 900


@dataclass
class AppConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)


OUTPUT_ROOT = "OutputVitalsMotion"
DATASET_ROOT = os.path.join(OUTPUT_ROOT, "dataset_master")
TRAINED_MODEL_DIR = "trained_models"
