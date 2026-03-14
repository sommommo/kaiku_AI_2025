import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class RuntimeState:
    ts: float = 0.0
    fps: float = 0.0
    frame_count: int = -1
    target_bin: int = -1
    bin_score: float = 0.0
    top_bins: List[Any] = field(default_factory=list)
    motion_level: float = 0.0
    motion_flag: bool = False
    stable_seconds: float = 0.0
    resp_rpm: Optional[float] = None
    heart_bpm: Optional[float] = None
    vitals_quality: str = "warming_up"
    model_label: str = "-"
    model_score: float = 0.0
    action_text: str = "等待資料"
    event_text: str = "-"
    vitals_alert_text: str = "normal"
    vitals_alert_level: int = 0
    resp_abnormal_flag: bool = False
    heart_abnormal_flag: bool = False
    range_profile: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    phase_wave: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    resp_wave: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    heart_wave: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    motion_wave: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    debug: Dict[str, Any] = field(default_factory=dict)


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = RuntimeState(ts=time.time())

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)
            self._state.ts = time.time()

    def snapshot(self) -> RuntimeState:
        with self._lock:
            s = self._state
            return RuntimeState(
                ts=s.ts,
                fps=s.fps,
                frame_count=s.frame_count,
                target_bin=s.target_bin,
                bin_score=s.bin_score,
                top_bins=list(s.top_bins),
                motion_level=s.motion_level,
                motion_flag=s.motion_flag,
                stable_seconds=s.stable_seconds,
                resp_rpm=s.resp_rpm,
                heart_bpm=s.heart_bpm,
                vitals_quality=s.vitals_quality,
                model_label=s.model_label,
                model_score=s.model_score,
                action_text=s.action_text,
                event_text=s.event_text,
                vitals_alert_text=s.vitals_alert_text,
                vitals_alert_level=s.vitals_alert_level,
                resp_abnormal_flag=s.resp_abnormal_flag,
                heart_abnormal_flag=s.heart_abnormal_flag,
                range_profile=np.array(s.range_profile, copy=True),
                phase_wave=np.array(s.phase_wave, copy=True),
                resp_wave=np.array(s.resp_wave, copy=True),
                heart_wave=np.array(s.heart_wave, copy=True),
                motion_wave=np.array(s.motion_wave, copy=True),
                debug=dict(s.debug),
            )
