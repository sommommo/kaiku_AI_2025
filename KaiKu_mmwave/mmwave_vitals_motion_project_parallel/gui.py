import os
import sys
import numpy as np

GUI_AVAILABLE = True
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
except Exception:
    GUI_AVAILABLE = False

from state import SharedState
from config import GUIConfig


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title: str, value: str = "--"):
        super().__init__()
        self.setObjectName("MetricCard")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("MetricTitle")
        self.value = QtWidgets.QLabel(value)
        self.value.setObjectName("MetricValue")
        lay.addWidget(self.title)
        lay.addWidget(self.value)

    def set_value(self, text: str):
        self.value.setText(text)


class LiveWindow(QtWidgets.QMainWindow):
    def __init__(self, shared: SharedState, cfg: GUIConfig):
        self.shared = shared
        self.cfg = cfg
        self.enabled = GUI_AVAILABLE
        if not self.enabled:
            print("[WARN] pyqtgraph / Qt not available, running headless.")
            return

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        super().__init__()
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self.setWindowTitle(cfg.title)
        self.resize(cfg.main_width, cfg.main_height)
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(int(cfg.refresh_ms))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(14)
        root.addLayout(top, 0)

        self.metric_panel = QtWidgets.QFrame()
        self.metric_panel.setObjectName("Panel")
        metric_layout = QtWidgets.QGridLayout(self.metric_panel)
        metric_layout.setContentsMargins(18, 18, 18, 18)
        metric_layout.setHorizontalSpacing(12)
        metric_layout.setVerticalSpacing(12)

        self.cards = {
            "range": MetricCard("Range / BIN", "--"),
            "score": MetricCard("SCORE", "--"),
            "motion": MetricCard("MOTION", "--"),
            "rpm": MetricCard("RPM", "--"),
            "bpm": MetricCard("BPM", "--"),
            "fps": MetricCard("FPS / Frame", "--"),
        }
        order = ["range", "score", "motion", "rpm", "bpm", "fps"]
        for i, key in enumerate(order):
            metric_layout.addWidget(self.cards[key], i // 2, i % 2)

        top.addWidget(self.metric_panel, 3)

        self.action_panel = QtWidgets.QFrame()
        self.action_panel.setObjectName("Panel")
        action_layout = QtWidgets.QVBoxLayout(self.action_panel)
        action_layout.setContentsMargins(18, 18, 18, 18)
        action_layout.setSpacing(8)

        title = QtWidgets.QLabel("現在的動作")
        title.setObjectName("PanelTitle")
        self.action_text = QtWidgets.QLabel("等待資料")
        self.action_text.setAlignment(QtCore.Qt.AlignCenter)
        self.action_text.setObjectName("ActionText")

        self.action_image = QtWidgets.QLabel("(動作圖)")
        self.action_image.setAlignment(QtCore.Qt.AlignCenter)
        self.action_image.setMinimumSize(240, 180)
        self.action_image.setObjectName("ImageBox")

        self.alert_title = QtWidgets.QLabel("呼吸 / 心率警示")
        self.alert_title.setObjectName("PanelTitle")
        self.alert_title.setStyleSheet("font-size:16px;")
        self.alert_text = QtWidgets.QLabel("normal")
        self.alert_text.setAlignment(QtCore.Qt.AlignCenter)
        self.alert_text.setObjectName("AlertText")
        self.alert_image = QtWidgets.QLabel("(警示圖)")
        self.alert_image.setAlignment(QtCore.Qt.AlignCenter)
        self.alert_image.setMinimumSize(240, 120)
        self.alert_image.setObjectName("AlertBox")

        self.action_hint = QtWidgets.QLabel("例如：normal / cough / fall / agitation")
        self.action_hint.setAlignment(QtCore.Qt.AlignCenter)
        self.action_hint.setObjectName("SubInfo")

        action_layout.addWidget(title)
        action_layout.addWidget(self.action_text)
        action_layout.addWidget(self.action_image, 1)
        action_layout.addWidget(self.alert_title)
        action_layout.addWidget(self.alert_text)
        action_layout.addWidget(self.alert_image)
        action_layout.addWidget(self.action_hint)
        top.addWidget(self.action_panel, 2)

        wave_panel = QtWidgets.QFrame()
        wave_panel.setObjectName("Panel")
        wave_layout = QtWidgets.QGridLayout(wave_panel)
        wave_layout.setContentsMargins(14, 14, 14, 14)
        wave_layout.setSpacing(10)
        root.addWidget(wave_panel, 1)

        self.range_plot, self.range_curve = self._make_plot("Range")
        self.range_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((30, 144, 255), width=2))
        self.range_plot.addItem(self.range_vline)
        self.phase_plot, self.phase_curve = self._make_plot("Phase")
        self.resp_plot, self.resp_curve = self._make_plot("Respiration")
        self.heart_plot, self.heart_curve = self._make_plot("Heart")
        self.motion_plot, self.motion_curve = self._make_plot("Motion")

        wave_layout.addWidget(self.range_plot, 0, 0)
        wave_layout.addWidget(self.phase_plot, 0, 1)
        wave_layout.addWidget(self.resp_plot, 1, 0)
        wave_layout.addWidget(self.heart_plot, 1, 1)
        wave_layout.addWidget(self.motion_plot, 2, 0, 1, 2)

        status = self.statusBar()
        self.status_label = QtWidgets.QLabel("MODEL: -- | EVENT: --")
        status.addPermanentWidget(self.status_label, 1)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6fb; font-family: Microsoft JhengHei, Arial; }
            QFrame#Panel { background: white; border: 2px solid #d9dfeb; border-radius: 18px; }
            QFrame#MetricCard { background: #f9fbff; border: 1px solid #dce5f2; border-radius: 14px; }
            QLabel#MetricTitle { color: #5b6577; font-size: 13px; }
            QLabel#MetricValue { color: #151b26; font-size: 24px; font-weight: 700; }
            QLabel#PanelTitle { color: #263042; font-size: 18px; font-weight: 700; }
            QLabel#ActionText { color: #0f6cbd; font-size: 28px; font-weight: 800; padding: 4px; }
            QLabel#AlertText { color: #c62828; font-size: 22px; font-weight: 800; padding: 2px; }
            QLabel#ImageBox { background: #f8fbff; border: 2px dashed #bfd7f5; border-radius: 16px; color: #5d6b82; font-size: 22px; }
            QLabel#AlertBox { background: #fff8f8; border: 2px dashed #f0b0b0; border-radius: 16px; color: #8a5a5a; font-size: 20px; }
            QLabel#SubInfo { color: #6a778d; font-size: 13px; }
        """)

    def _make_plot(self, title: str):
        pw = pg.PlotWidget(title=title)
        pw.showGrid(x=True, y=True, alpha=0.18)
        pw.getPlotItem().setContentsMargins(8, 8, 8, 8)
        pw.setMenuEnabled(False)
        pw.setClipToView(True)
        pw.setDownsampling(mode="peak")
        curve = pw.plot(pen=pg.mkPen(width=2))
        return pw, curve

    def _tail(self, arr):
        arr = np.asarray(arr)
        return arr[-self.cfg.max_points:] if arr.size else arr

    def _set_image(self, widget, base_dir: str, label: str, fallback: str):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = os.path.join(base_dir, f"{label}{ext}")
            if os.path.exists(p):
                pix = QtGui.QPixmap(p)
                if not pix.isNull():
                    widget.setPixmap(pix.scaled(
                        widget.width() - 10,
                        widget.height() - 10,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    ))
                    widget.setText("")
                    return
        widget.setPixmap(QtGui.QPixmap())
        widget.setText(label if label not in ("", "-") else fallback)

    def refresh(self):
        s = self.shared.snapshot()
        self.cards["range"].set_value("--" if s.target_bin < 0 else f"BIN {s.target_bin}")
        self.cards["score"].set_value(f"{s.bin_score:.3f}")
        self.cards["motion"].set_value(f"{s.motion_level:.3f}")
        rpm_text = "--" if s.resp_rpm is None else f"{s.resp_rpm:.1f}"
        bpm_text = "--" if s.heart_bpm is None else f"{s.heart_bpm:.1f}"
        if s.vitals_quality == "motion_interference":
            rpm_text += " *"
            bpm_text += " *"
        self.cards["rpm"].set_value(rpm_text)
        self.cards["bpm"].set_value(bpm_text)
        frame_text = "--" if s.frame_count < 0 else str(int(s.frame_count))
        self.cards["fps"].set_value(f"{s.fps:.2f} / {frame_text}")


        action_label = s.action_text or s.model_label or "等待資料"
        self.action_text.setText(action_label)
        hint_event = s.event_text if s.event_text not in (None, "") else "-"
        quality_map = {"stable": "Vitals stable", "motion_interference": "Vitals with motion", "warming_up": "Vitals warming up"}
        self.action_hint.setText(f"MODEL SCORE: {s.model_score:.2f}   |   EVENT: {hint_event}   |   {quality_map.get(s.vitals_quality, s.vitals_quality)}")
        self._set_image(self.action_image, self.cfg.action_asset_dir, action_label.split("/")[-1], "(動作圖)")

        alert_label_map = {
            "normal": "正常",
            "watch": "監測中",
            "resp_alert": "呼吸異常",
            "heart_alert": "心率異常",
            "resp_heart_alert": "呼吸&心率異常",
        }
        alert_text = alert_label_map.get(s.vitals_alert_text, s.vitals_alert_text)
        if s.resp_abnormal_flag or s.heart_abnormal_flag:
            flags = []
            if s.resp_abnormal_flag:
                flags.append("Resp")
            if s.heart_abnormal_flag:
                flags.append("Heart")
            alert_text += f"  ({'/'.join(flags)})"
        self.alert_text.setText(alert_text)
        self._set_image(self.alert_image, self.cfg.alert_asset_dir, s.vitals_alert_text, "(警示圖)")

        if s.range_profile.size:
            self.range_curve.setData(s.range_profile)
            if s.target_bin >= 0:
                self.range_vline.setValue(s.target_bin)
        self.phase_curve.setData(self._tail(s.phase_wave))
        self.resp_curve.setData(self._tail(s.resp_wave))
        self.heart_curve.setData(self._tail(s.heart_wave))
        self.motion_curve.setData(self._tail(s.motion_wave))

        event_text = s.event_text if s.event_text not in (None, "") else "-"
        self.status_label.setText(
            f"MODEL: {s.model_label} ({s.model_score:.2f})   |   MOTION FLAG: {'Y' if s.motion_flag else 'N'}   |   VITALS: {s.vitals_quality}   |   ALERT: {s.vitals_alert_text}   |   EVENT: {event_text}"
        )
        self.app.processEvents()

    def run(self):
        if not self.enabled:
            return 0
        self.show()
        return self.app.exec_()


def run_gui(shared: SharedState, cfg: GUIConfig):
    if GUI_AVAILABLE and QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv)
    return LiveWindow(shared, cfg).run()
