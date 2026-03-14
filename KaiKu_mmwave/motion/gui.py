import numpy as np

GUI_AVAILABLE = True
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets
except Exception:
    GUI_AVAILABLE = False


class LiveWindow:
    def __init__(self, max_gui_points: int = 800):
        self.max_gui_points = max_gui_points
        if not GUI_AVAILABLE:
            self.enabled = False
            print("[WARN] pyqtgraph / Qt not available, running headless.")
            return

        self.enabled = True
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="mmWave Research System")
        self.win.resize(1300, 900)
        self.win.show()

        self.range_plot = self.win.addPlot(title="Range Profile Score")
        self.range_curve = self.range_plot.plot()
        self.range_vline = pg.InfiniteLine(angle=90, movable=False)
        self.range_plot.addItem(self.range_vline)
        self.win.nextRow()

        self.phase_plot = self.win.addPlot(title="Phase (target bin)")
        self.phase_curve = self.phase_plot.plot()
        self.win.nextRow()

        self.resp_plot = self.win.addPlot(title="Respiration Signal")
        self.resp_curve = self.resp_plot.plot()
        self.win.nextRow()

        self.heart_plot = self.win.addPlot(title="Heart Signal")
        self.heart_curve = self.heart_plot.plot()
        self.win.nextRow()

        self.motion_plot = self.win.addPlot(title="Motion Energy")
        self.motion_curve = self.motion_plot.plot()

        self.label = QtWidgets.QLabel()
        self.label.setStyleSheet("font-size: 16px; padding: 8px;")
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(self.label)
        self.win.scene().addItem(proxy)
        proxy.setPos(10, 10)

    def update(self, summary, system):
        if not self.enabled or summary is None:
            return

        range_score = np.asarray(summary.range_score)
        if range_score.size > 0:
            self.range_curve.setData(range_score)
            if summary.target_bin is not None:
                self.range_vline.setValue(summary.target_bin)

        self.phase_curve.setData(summary.phase_series[-self.max_gui_points:] if summary.phase_series.size > 0 else [])
        self.resp_curve.setData(summary.resp_signal[-self.max_gui_points:] if summary.resp_signal.size > 0 else [])
        self.heart_curve.setData(summary.heart_signal[-self.max_gui_points:] if summary.heart_signal.size > 0 else [])

        motion = system.motion_energy.as_array()
        self.motion_curve.setData(motion[-self.max_gui_points:] if motion.size > 0 else [])

        last_event = system.events[-1]["label"] if system.events else "-"
        txt = (
            f"FPS: {summary.fps:.2f} | Target bin: {summary.target_bin} | "
            f"Motion: {summary.motion:.3f} (z={summary.motion_z:.2f}) | "
            f"Resp: {summary.rpm} RPM | Heart: {summary.bpm} BPM | "
            f"Bursts: {summary.cough_bursts} | Last event: {last_event}"
        )
        self.label.setText(txt)
        self.app.processEvents()
