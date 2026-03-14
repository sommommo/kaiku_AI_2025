import json
import time
from typing import Dict, Any

import numpy as np

from config import SystemConfig, SessionPaths
from dataset_tools import DatasetRecorder
from processing import FeatureExtractor, Summary
from utils import RingArray


class RadarSystem:
    def __init__(self, cfg: SystemConfig, paths: SessionPaths):
        self.cfg = cfg
        self.paths = paths
        self.extractor = FeatureExtractor(cfg)
        self.dataset_recorder = DatasetRecorder(cfg, paths)

        self.raw_frames = RingArray(cfg.frame_buffer_max)
        self.frame_counts = RingArray(cfg.frame_buffer_max)
        self.timestamps = RingArray(cfg.frame_buffer_max)
        self.motion_energy = RingArray(cfg.frame_buffer_max)
        self.target_bin_hist = RingArray(cfg.frame_buffer_max)
        self.phase_hist = RingArray(cfg.frame_buffer_max)

        self.events = []
        self.last_event_time = {"fall": 0.0, "cough": 0.0}

    def estimate_fps(self) -> float:
        ts = self.timestamps.as_array()
        if ts.size >= 20:
            dt = np.diff(ts)
            dt = dt[dt > 1e-4]
            if dt.size > 0:
                return float(1.0 / np.median(dt))
        return self.cfg.fps_est

    def process_frame(self, raw: np.ndarray, frame_count: int = -1) -> Summary:
        now = time.time()
        raw = np.asarray(raw)
        if raw.ndim != 3:
            raise ValueError(f"Expected raw ndim=3, got shape={raw.shape}")

        self.extractor.set_shape_if_needed(raw)

        self.raw_frames.append(raw.astype(np.int16, copy=False) if raw.dtype != np.int16 else raw)
        self.frame_counts.append(frame_count)
        self.timestamps.append(now)

        motion = self.extractor.compute_motion_energy(raw)
        self.motion_energy.append(motion)

        fft_frame = self.extractor.range_fft(raw)
        score = self.extractor.range_profile_score(fft_frame)
        target_bin = self.extractor.pick_target_bin(score, now)
        self.target_bin_hist.append(-1 if target_bin is None else int(target_bin))

        phase_value = np.nan
        if target_bin is not None:
            phase_value = self.extractor.phase_from_fft_frame(fft_frame, target_bin)
        self.phase_hist.append(phase_value)

        summary = self.extractor.build_summary(
            fps=self.estimate_fps(),
            phase_hist=self.phase_hist.as_array(),
            motion_hist=self.motion_energy.as_array(),
            score=score,
            motion=motion,
        )
        self.detect_events(summary)
        return summary

    def detect_events(self, summary: Summary):
        now = time.time()
        fps = summary.fps
        motion_series = self.motion_energy.as_array().astype(np.float64)
        if motion_series.size < max(10, int(2 * fps)):
            return

        if summary.motion_z > self.cfg.motion_z_fall and (now - self.last_event_time["fall"]) > 3.0:
            still_n = max(3, int(self.cfg.fall_still_sec * fps))
            recent = motion_series[-still_n:]
            med = np.median(recent)
            mad = np.median(np.abs(recent - med)) + 1e-9
            rz = np.abs(0.6745 * (recent - med) / mad)
            if np.mean(rz) < self.cfg.low_motion_z:
                self.raise_event("fall", summary)
                self.last_event_time["fall"] = now

        if summary.motion_z > self.cfg.motion_z_cough and (now - self.last_event_time["cough"]) > 1.5:
            if summary.cough_bursts >= self.cfg.cough_min_bursts:
                self.raise_event("cough", summary)
                self.last_event_time["cough"] = now

    def raise_event(self, label: str, summary: Summary):
        event = {
            "time": time.time(),
            "label": label,
            "target_bin": summary.target_bin,
            "motion": summary.motion,
            "motion_z": summary.motion_z,
            "rpm": summary.rpm,
            "bpm": summary.bpm,
        }
        self.events.append(event)
        print(f"[EVENT] {label} | bin={event['target_bin']} motion={event['motion']:.3f} z={event['motion_z']:.2f}")
        if self.cfg.enable_dataset_recorder:
            self.save_segment(label)

    def save_segment(self, label: str):
        fps = self.estimate_fps()
        pre_n = int(self.cfg.event_pre_sec * fps)
        post_n = int(self.cfg.event_post_sec * fps)

        raw_arr = self.raw_frames.as_array()
        ts_arr = self.timestamps.as_array()
        fc_arr = self.frame_counts.as_array()
        motion_arr = self.motion_energy.as_array()
        bin_arr = self.target_bin_hist.as_array()
        phase_arr = self.extractor.phase_series_from_history(self.phase_hist.as_array())

        if raw_arr.shape[0] < (pre_n + 4):
            return

        end_idx = raw_arr.shape[0]
        start_idx = max(0, end_idx - (pre_n + post_n))

        self.dataset_recorder.write_segment(
            label=label,
            session_name=self.paths.session_name,
            raw=raw_arr[start_idx:end_idx],
            timestamps=ts_arr[start_idx:end_idx],
            frame_count=fc_arr[start_idx:end_idx],
            motion=motion_arr[start_idx:end_idx],
            target_bin=bin_arr[start_idx:end_idx],
            phase=phase_arr[start_idx:end_idx] if phase_arr.size > 0 else np.array([]),
            start_idx=start_idx,
            end_idx=end_idx,
            fps=fps,
            extra_meta={"setting_dir": self.cfg.setting_dir},
        )

    def save_session(self):
        raw_arr = self.raw_frames.as_array()
        fc_arr = self.frame_counts.as_array()
        ts_arr = self.timestamps.as_array()
        motion_arr = self.motion_energy.as_array()
        bin_arr = self.target_bin_hist.as_array()
        phase_arr = self.extractor.phase_series_from_history(self.phase_hist.as_array())

        np.savez_compressed(
            self.paths.raw_save_path,
            raw=raw_arr,
            frame_count=fc_arr,
            timestamps=ts_arr,
            motion=motion_arr,
            target_bin=bin_arr,
            phase=phase_arr,
            setting_dir=self.cfg.setting_dir,
        )

        meta: Dict[str, Any] = {
            "setting_dir": self.cfg.setting_dir,
            "session_dir": self.paths.session_dir,
            "dataset_root": self.paths.dataset_root,
            "dataset_index_path": self.paths.dataset_index_path,
            "fps_est": self.estimate_fps(),
            "rx_count": self.extractor.rx_count,
            "chirp_count": self.extractor.chirp_count,
            "sample_count": self.extractor.sample_count,
            "events": self.events,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.paths.meta_save_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[SAVE] session raw ->", self.paths.raw_save_path)
        print("[SAVE] session meta ->", self.paths.meta_save_path)
        print("[SAVE] dataset index ->", self.paths.dataset_index_path)
