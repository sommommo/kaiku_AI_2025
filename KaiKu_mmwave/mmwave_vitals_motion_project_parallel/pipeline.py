from collections import deque
import time
import numpy as np

from config import DSPConfig, ModelConfig
from state import SharedState
from dsp import RangeFFT, coherent_bin_vector, combine_rx, robust_unwrap_append
from bin_tracker import ResearchBinTracker
from motion_analyzer import MotionSuppressor
from vitals import ResearchVitalsEstimator
from model_runner import MotionModelRunner


class MmWaveVitalsMotionPipeline:
    def __init__(self, shared: SharedState, dsp_cfg: DSPConfig, model_cfg: ModelConfig):
        self.shared = shared
        self.cfg = dsp_cfg
        self.model_cfg = model_cfg
        self.rangefft = RangeFFT(dsp_cfg.keep_samples, dsp_cfg.fft_bins)
        self.fps = float(dsp_cfg.fps_init)
        self.bin_tracker = ResearchBinTracker(
            bins=dsp_cfg.fft_bins,
            ignore_bins=dsp_cfg.ignore_bins,
            fps=self.fps,
            reselect_every_sec=dsp_cfg.reselect_every_sec,
            hold_sec=dsp_cfg.bin_hold_sec,
            switch_margin=dsp_cfg.switch_margin,
            energy_weight=dsp_cfg.energy_weight,
            phase_snr_weight=dsp_cfg.phase_snr_weight,
        )
        self.motion = MotionSuppressor(
            fs=self.fps,
            short_sec=dsp_cfg.motion_short_sec,
            phase_std_thr=dsp_cfg.motion_thr_phase_std,
            bin_jump_thr=dsp_cfg.motion_thr_bin_jump,
            freeze_sec=dsp_cfg.motion_freeze_sec,
        )
        self.vitals = ResearchVitalsEstimator(
            fs=self.fps,
            window_sec=dsp_cfg.window_sec,
            estimate_min_ratio=dsp_cfg.estimate_min_ratio,
            resp_band=dsp_cfg.resp_band,
            heart_band=dsp_cfg.heart_band,
        )
        self.model = MotionModelRunner(model_cfg.model_path, model_cfg.label_map, model_cfg.score_threshold) if model_cfg.enabled else MotionModelRunner(None, model_cfg.label_map)

        self._t0 = None
        self._fc0 = None
        self._last_phase_unwrapped = None
        self._phase_wave = deque(maxlen=max(64, int(dsp_cfg.window_sec * self.fps)))
        self._motion_wave = deque(maxlen=max(64, int(dsp_cfg.window_sec * self.fps)))
        self._last_infer_t = 0.0
        self._last_action_text = "等待資料"

    def _update_fps(self, fc: int, t_now: float):
        if fc < 0:
            return
        if self._t0 is None:
            self._t0 = t_now
            self._fc0 = fc
            return
        dt = t_now - self._t0
        df = fc - self._fc0
        if dt >= 2.0 and df > 0:
            new_fps = float(np.clip(df / dt, 5.0, 60.0))
            self.fps = 0.8 * self.fps + 0.2 * new_fps
            self.bin_tracker.set_fps(self.fps)
            self.motion.set_fps(self.fps, self.cfg.motion_short_sec, self.cfg.motion_freeze_sec)
            self.vitals.set_fps(self.fps, self.cfg.window_sec)
            self._phase_wave = deque(self._phase_wave, maxlen=max(64, int(self.cfg.window_sec * self.fps)))
            self._motion_wave = deque(self._motion_wave, maxlen=max(64, int(self.cfg.window_sec * self.fps)))
            self._t0 = t_now
            self._fc0 = fc

    def _predict_action(self, t_now: float, motion_level: float, range_profile: np.ndarray, target_bin: int, rpm, bpm):
        gate = float(self.model_cfg.motion_gate_threshold)
        if motion_level < gate:
            label = self.model_cfg.idle_label
            return label, 1.0, "-", "motion_gate_idle"

        if not self.model.enabled:
            label = "偵測到動作"
            score = max(0.0, min(1.0, motion_level / max(gate, 1e-6)))
            return label, score, "motion_only", "model_not_loaded"

        min_len = max(8, int(self.model_cfg.infer_window_sec * self.fps))
        stride = float(self.model_cfg.infer_stride_sec)
        if len(self._phase_wave) < min_len:
            waiting_label = self._last_action_text if self.model.enabled else "蒐集中"
            return waiting_label, 0.0, "-", "waiting_window"
        if (t_now - self._last_infer_t) < stride:
            snap = self.shared.snapshot()
            return snap.action_text, snap.model_score, snap.event_text, "stride_hold"

        self._last_infer_t = t_now
        phase_arr = np.asarray(self._phase_wave, dtype=np.float32)[-min_len:]
        motion_arr = np.asarray(self._motion_wave, dtype=np.float32)[-min_len:]
        feat = self.model.build_feature_vector(phase_arr, motion_arr, range_profile, self.fps, rpm, bpm, target_bin)
        pred = self.model.predict(feat)

        label = pred["label"]
        score = float(pred["score"])
        confident = bool(pred.get("is_confident", False))

        if not confident:
            action = "疑似動作"
            event = "-"
        elif label == self.model_cfg.idle_label:
            action = self.model_cfg.idle_label
            event = "-"
        else:
            action = label
            event = label

        self._last_action_text = action
        return action, score, event, "model_predict"

    def push_frame(self, raw: np.ndarray, fc: int, t_now: float):
        self._update_fps(fc, t_now)
        F = self.rangefft.run(raw)
        self.bin_tracker.update(F)
        target_bin, score, top_bins = self.bin_tracker.maybe_pick(t_now)
        range_profile = self.bin_tracker.energy.astype(np.float32) if hasattr(self.bin_tracker, "energy") else np.array([], dtype=np.float32)

        if target_bin < 0:
            self.shared.update(frame_count=fc, top_bins=top_bins, range_profile=range_profile)
            return

        rx_vec = coherent_bin_vector(F, target_bin)
        z = combine_rx(rx_vec, self.cfg.rx_mode)
        phase_angle = float(np.angle(z))
        phase_unwrapped = phase_angle if self._last_phase_unwrapped is None else robust_unwrap_append(self._last_phase_unwrapped, phase_angle)
        self._last_phase_unwrapped = phase_unwrapped

        motion_level, motion_flag, stable_sec, phase_std, motion_z = self.motion.update(phase_unwrapped, self.bin_tracker.bin_jump)
        self._phase_wave.append(phase_unwrapped)
        self._motion_wave.append(motion_level)

        self.vitals.update(phase_unwrapped)
        rpm, bpm, resp_wave, heart_wave, dbg_v = self.vitals.estimate()
        if dbg_v.get("buf_len", 0) < dbg_v.get("need_len", 0):
            vitals_quality = "warming_up"
        elif motion_flag:
            vitals_quality = "motion_interference"
        else:
            vitals_quality = "stable"

        action_text, model_score, event_text, action_reason = self._predict_action(
            t_now=t_now,
            motion_level=motion_level,
            range_profile=range_profile,
            target_bin=target_bin,
            rpm=rpm,
            bpm=bpm,
        )
        model_label = action_text

        debug = {
            "phase_angle": round(phase_angle, 5),
            "phase_unwrapped": round(phase_unwrapped, 5),
            "phase_std": round(phase_std, 5),
            "motion_z": round(motion_z, 5),
            "bin_jump": float(self.bin_tracker.bin_jump),
            "action_reason": action_reason,
            "vitals_quality": vitals_quality,
            **dbg_v,
        }
        self.shared.update(
            frame_count=fc,
            fps=self.fps,
            target_bin=int(target_bin),
            bin_score=float(score),
            top_bins=top_bins,
            motion_level=float(motion_level),
            motion_flag=bool(motion_flag),
            stable_seconds=float(stable_sec),
            resp_rpm=rpm,
            heart_bpm=bpm,
            vitals_quality=vitals_quality,
            model_label=model_label,
            model_score=float(model_score),
            action_text=action_text,
            event_text=event_text or "-",
            range_profile=range_profile,
            phase_wave=np.asarray(self._phase_wave, dtype=np.float32),
            resp_wave=np.asarray(resp_wave, dtype=np.float32),
            heart_wave=np.asarray(heart_wave, dtype=np.float32),
            motion_wave=np.asarray(self._motion_wave, dtype=np.float32),
            debug=debug,
        )
