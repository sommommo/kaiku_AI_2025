import time
from dataclasses import dataclass
from config import AlertConfig


@dataclass
class AlertResult:
    label: str
    level: int
    resp_abnormal: bool
    heart_abnormal: bool
    debug: dict


class VitalsAlertEvaluator:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.resp_since = None
        self.heart_since = None
        self.last_alert_t = 0.0

    def _update_since(self, now: float, value, low: float, high: float, prev_since):
        if value is None:
            return None, False
        abnormal = value < low or value > high
        if abnormal:
            if prev_since is None:
                prev_since = now
        else:
            prev_since = None
        return prev_since, abnormal

    def update(self, rpm, bpm) -> AlertResult:
        now = time.time()
        self.resp_since, resp_abnormal = self._update_since(now, rpm, self.cfg.resp_low_rpm, self.cfg.resp_high_rpm, self.resp_since)
        self.heart_since, heart_abnormal = self._update_since(now, bpm, self.cfg.heart_low_bpm, self.cfg.heart_high_bpm, self.heart_since)

        resp_dur = 0.0 if self.resp_since is None else now - self.resp_since
        heart_dur = 0.0 if self.heart_since is None else now - self.heart_since
        sustained_resp = resp_abnormal and resp_dur >= self.cfg.sustain_sec
        sustained_heart = heart_abnormal and heart_dur >= self.cfg.sustain_sec

        label = "normal"
        level = 0
        if sustained_resp and sustained_heart and self.cfg.show_combined_first:
            label = "resp_heart_alert"
            level = 2
        elif sustained_resp:
            label = "resp_alert"
            level = 1
        elif sustained_heart:
            label = "heart_alert"
            level = 1
        elif resp_abnormal or heart_abnormal:
            label = "watch"
            level = 0

        if level > 0:
            self.last_alert_t = now
        elif (now - self.last_alert_t) < self.cfg.cooldown_sec:
            if resp_abnormal and heart_abnormal and self.cfg.show_combined_first:
                label = "resp_heart_alert"
                level = 2
            elif resp_abnormal:
                label = "resp_alert"
                level = 1
            elif heart_abnormal:
                label = "heart_alert"
                level = 1

        return AlertResult(
            label=label,
            level=level,
            resp_abnormal=bool(resp_abnormal),
            heart_abnormal=bool(heart_abnormal),
            debug={
                "resp_low_rpm": self.cfg.resp_low_rpm,
                "resp_high_rpm": self.cfg.resp_high_rpm,
                "heart_low_bpm": self.cfg.heart_low_bpm,
                "heart_high_bpm": self.cfg.heart_high_bpm,
                "resp_abnormal_dur": round(resp_dur, 2),
                "heart_abnormal_dur": round(heart_dur, 2),
                "sustain_sec": self.cfg.sustain_sec,
            },
        )
