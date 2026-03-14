import threading
import time
import traceback

from config import AppConfig
from state import SharedState
from pipeline import MmWaveVitalsMotionPipeline
from gui import run_gui
from device_kkt import FRMDevice, RealTimeUpdater


def run_device(shared: SharedState, stop_evt: threading.Event, cfg: AppConfig):
    device = None
    try:
        pipeline = MmWaveVitalsMotionPipeline(shared, cfg.dsp, cfg.model, cfg.alert)
        updater = RealTimeUpdater(pipeline)
        device = FRMDevice(cfg.device.setting_dir, cfg.device.stream_type)
        device.start(updater)
        while not stop_evt.is_set():
            time.sleep(0.1)
    except Exception:
        traceback.print_exc()
    finally:
        if device is not None:
            try:
                device.stop()
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    cfg = AppConfig()
    # cfg.model.model_path = r"trained_models/mmwave_action_rf.joblib"
    # cfg.model.motion_gate_threshold = 3.0
    # cfg.alert.sustain_sec = 8.0

    shared = SharedState()
    stop_evt = threading.Event()
    worker = threading.Thread(target=run_device, args=(shared, stop_evt, cfg), daemon=False)
    worker.start()
    try:
        run_gui(shared, cfg.gui)
    finally:
        stop_evt.set()
        worker.join(timeout=3.0)
