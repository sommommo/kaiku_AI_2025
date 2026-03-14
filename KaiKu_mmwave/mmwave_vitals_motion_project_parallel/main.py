import threading
import time

from config import AppConfig
from state import SharedState
from pipeline import MmWaveVitalsMotionPipeline
from gui import run_gui
from device_kkt import FRMDevice, RealTimeUpdater


def run_device(shared: SharedState, stop_evt: threading.Event, cfg: AppConfig):
    pipeline = MmWaveVitalsMotionPipeline(shared, cfg.dsp, cfg.model)
    updater = RealTimeUpdater(pipeline)
    device = FRMDevice(cfg.device.setting_dir, cfg.device.stream_type)
    device.start(updater)
    try:
        while not stop_evt.is_set():
            time.sleep(0.1)
    finally:
        device.stop()


if __name__ == "__main__":
    cfg = AppConfig()
    # 例：
    # 請指定已訓練模型；未指定時會只用 MOTION 顯示「偵測到動作」，不做分類
    # cfg.model.model_path = r"trained_models/mmwave_action_rf.joblib"
    # cfg.model.motion_gate_threshold = 0.35

    shared = SharedState()
    stop_evt = threading.Event()
    worker = threading.Thread(target=run_device, args=(shared, stop_evt, cfg), daemon=True)
    worker.start()
    try:
        run_gui(shared, cfg.gui)
    finally:
        stop_evt.set()
        worker.join(timeout=2.0)
