import logging
import time
import numpy as np

try:
    from KKT_Module import kgl
    from KKT_Module.FiniteReceiverMachine import FRM
    from KKT_Module.GuiUpdater.GuiUpdater import Updater
    from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
    from KKT_Module.DataReceive.Core import Results
    from KKT_Module.SettingProcess.SettingProccess import SettingProc
    try:
        from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
    except Exception:
        from KKT_Module.Configs import SettingConfigs
    KKT_AVAILABLE = True
except Exception:
    KKT_AVAILABLE = False
    class Updater:
        pass
    class Results:
        pass


def connect_device_like_gui():
    dev = kgl.ksoclib.connectDevice()
    if dev == "Unknow":
        raise RuntimeError("Unknown device. Please reconnect and try again.")


def run_setting_script(setting_dir: str):
    ksp = SettingProc()
    cfg = SettingConfigs()
    try:
        cfg.Chip_ID = kgl.ksoclib.getChipID().split(" ")[0]
    except Exception:
        pass
    cfg.Processes = [
        "Reset Device",
        "Gen Process Script",
        "Gen Param Dict",
        "Get Gesture Dict",
        "Set Script",
        "Run SIC",
        "Phase Calibration",
        "Modulation On",
    ]
    cfg.setScriptDir(setting_dir)
    ksp.startUp(cfg)


def set_properties(obj: object, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


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


class RealTimeUpdater(Updater):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.last_fc = None

    def update(self, res: Results):
        try:
            raw = np.asarray(res["raw_data"].data)
        except Exception:
            return
        fc = -1
        try:
            fc_obj = res.get("frame_count", None)
            if fc_obj is not None:
                fc = to_int_safe(fc_obj, default=-1)
        except Exception:
            pass
        if fc != -1 and self.last_fc is not None and fc == self.last_fc:
            return
        if fc != -1:
            self.last_fc = fc
        self.pipeline.push_frame(raw, fc, time.time())


class FRMDevice:
    def __init__(self, setting_dir: str, stream_type: str = "raw_data"):
        self.setting_dir = setting_dir
        self.stream_type = stream_type
        if not KKT_AVAILABLE:
            raise ImportError("KKT_Module not installed. Please copy this project into your radar PC environment.")

    def start(self, updater: Updater):
        logging.disable(logging.CRITICAL)
        kgl.setLib()
        connect_device_like_gui()
        run_setting_script(self.setting_dir)

        if self.stream_type == "raw_data":
            kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
        else:
            kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

        receiver = MultiResult4168BReceiver()
        set_properties(receiver, actions=1, rbank_ch_enable=7, read_interrupt=0, clear_interrupt=0)
        FRM.setReceiver(receiver)
        FRM.setUpdater(updater)
        FRM.trigger()
        FRM.start()

    def stop(self):
        try:
            FRM.stop()
        except Exception:
            pass
        try:
            kgl.ksoclib.closeDevice()
        except Exception:
            pass
