import time
import numpy as np

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

from config import SystemConfig
from utils import to_int_safe


def connect_device_like_gui():
    dev = kgl.ksoclib.connectDevice()
    if dev == "Unknow":
        raise RuntimeError("Unknown device. Please reconnect and try again.")


def run_setting_script(setting_name: str):
    ksp = SettingProc()
    cfg = SettingConfigs()
    try:
        cfg.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    except Exception:
        pass

    cfg.Processes = [
        'Reset Device',
        'Gen Process Script',
        'Gen Param Dict', 'Get Gesture Dict',
        'Set Script',
        'Run SIC',
        'Phase Calibration',
        'Modulation On'
    ]
    cfg.setScriptDir(f'{setting_name}')
    ksp.startUp(cfg)


def set_properties(obj: object, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


class ResearchUpdater(Updater):
    def __init__(self, system, window=None):
        super().__init__()
        self.system = system
        self.window = window
        self.last_fc = None
        self.frame_calls = 0

    def update(self, res: Results):
        self.frame_calls += 1
        try:
            raw_obj = res['raw_data']
            raw = np.asarray(raw_obj.data)
        except Exception:
            return

        fc = -1
        try:
            fc_obj = res.get('frame_count', None)
            if fc_obj is not None:
                fc = to_int_safe(fc_obj, default=-1)
        except Exception:
            pass

        if fc != -1 and self.last_fc is not None and fc == self.last_fc:
            return
        self.last_fc = fc if fc != -1 else self.last_fc

        summary = self.system.process_frame(raw, frame_count=fc)
        if self.window is not None:
            self.window.update(summary, self.system)

        if self.frame_calls % 30 == 0:
            print(
                f"[live] fps={summary.fps:.2f} bin={summary.target_bin} "
                f"motion={summary.motion:.3f} z={summary.motion_z:.2f} "
                f"resp={summary.rpm} heart={summary.bpm}"
            )


class KKTRunner:
    def __init__(self, cfg: SystemConfig, system, window=None):
        self.cfg = cfg
        self.system = system
        self.window = window

    def start(self):
        kgl.setLib()
        connect_device_like_gui()
        run_setting_script(self.cfg.setting_dir)

        if self.cfg.stream_type == "raw_data":
            kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
        else:
            kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

        receiver = MultiResult4168BReceiver()
        set_properties(receiver, actions=1, rbank_ch_enable=7, read_interrupt=0, clear_interrupt=0)

        updater = ResearchUpdater(system=self.system, window=self.window)
        FRM.setReceiver(receiver)
        FRM.setUpdater(updater)
        FRM.trigger()
        FRM.start()
        print("[INFO] FRM started. Live research system running...")

    def loop(self):
        while True:
            time.sleep(0.05)
            if self.window is not None and self.window.enabled:
                self.window.app.processEvents()

    def stop(self):
        try:
            FRM.stop()
        except Exception:
            pass
        try:
            kgl.ksoclib.closeDevice()
        except Exception:
            pass
