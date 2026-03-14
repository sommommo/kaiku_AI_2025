import os
import csv
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from config import SystemConfig, SessionPaths
from utils import SignalTools


LABEL_TO_ID: Dict[str, int] = {
    "fall": 0,
    "cough": 1,
    "sit": 2,
    "normal": 3,
    "bend": 4,
    "walk": 5,
    "other": 6,
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

INDEX_HEADERS = [
    "segment_path",
    "label",
    "label_id",
    "session_name",
    "start_idx",
    "end_idx",
    "num_frames",
    "fps",
    "target_bin_mean",
    "motion_mean",
    "motion_std",
    "phase_std",
    "rpm",
    "bpm",
    "created_at",
]


def ensure_index_csv(paths: SessionPaths) -> str:
    index_path = paths.dataset_index_path
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    if not os.path.exists(index_path):
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(INDEX_HEADERS)
    return index_path


@dataclass
class SegmentStats:
    fps: float
    target_bin_mean: float
    motion_mean: float
    motion_std: float
    phase_std: float
    rpm: Optional[float]
    bpm: Optional[float]


class DatasetRecorder:
    def __init__(self, cfg: SystemConfig, paths: SessionPaths):
        self.cfg = cfg
        self.paths = paths
        self.index_path = ensure_index_csv(paths)

    def _segment_stats(self, motion: np.ndarray, target_bin: np.ndarray, phase: np.ndarray, fps: float) -> SegmentStats:
        motion = np.asarray(motion, dtype=np.float64)
        target_bin = np.asarray(target_bin, dtype=np.float64)
        phase = np.asarray(phase, dtype=np.float64)

        rpm = None
        bpm = None
        if phase.size >= max(32, int(4 * fps)):
            ph = SignalTools.detrend(phase)
            resp = SignalTools.bandpass_fft(ph, fps, *self.cfg.resp_band)
            heart = SignalTools.bandpass_fft(ph, fps, *self.cfg.heart_band)
            f_r, _ = SignalTools.dominant_freq(resp, fps, *self.cfg.resp_band)
            f_h, _ = SignalTools.dominant_freq(heart, fps, *self.cfg.heart_band)
            rpm = None if f_r is None else f_r * 60.0
            bpm = None if f_h is None else f_h * 60.0

        return SegmentStats(
            fps=float(fps),
            target_bin_mean=float(np.nanmean(target_bin)) if target_bin.size > 0 else -1.0,
            motion_mean=float(np.mean(motion)) if motion.size > 0 else 0.0,
            motion_std=float(np.std(motion)) if motion.size > 0 else 0.0,
            phase_std=float(np.std(phase)) if phase.size > 0 else 0.0,
            rpm=rpm,
            bpm=bpm,
        )

    def write_segment(
        self,
        label: str,
        session_name: str,
        raw: np.ndarray,
        timestamps: np.ndarray,
        frame_count: np.ndarray,
        motion: np.ndarray,
        target_bin: np.ndarray,
        phase: np.ndarray,
        start_idx: int,
        end_idx: int,
        fps: float,
        extra_meta: Optional[dict] = None,
    ) -> str:
        if label not in LABEL_TO_ID:
            label = "other"

        label_dir = os.path.join(self.paths.dataset_label_root, label)
        os.makedirs(label_dir, exist_ok=True)
        base = time.strftime(f"{label}_%Y%m%d_%H%M%S")
        path = os.path.join(label_dir, base + ".npz")

        stats = self._segment_stats(motion, target_bin, phase, fps)
        meta = {
            "label": label,
            "label_id": LABEL_TO_ID[label],
            "session_name": session_name,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "num_frames": int(raw.shape[0]),
            "fps": float(fps),
            "target_bin_mean": stats.target_bin_mean,
            "motion_mean": stats.motion_mean,
            "motion_std": stats.motion_std,
            "phase_std": stats.phase_std,
            "rpm": stats.rpm,
            "bpm": stats.bpm,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra_meta:
            meta.update(extra_meta)

        np.savez_compressed(
            path,
            raw=raw,
            timestamps=timestamps,
            frame_count=frame_count,
            motion=motion,
            target_bin=target_bin,
            phase=phase,
            label=label,
            label_id=LABEL_TO_ID[label],
            meta=json.dumps(meta, ensure_ascii=False),
        )
        self._append_index(path, meta)
        print(f"[DATASET] saved -> {path}")
        return path

    def _append_index(self, segment_path: str, meta: dict):
        row = [
            segment_path,
            meta["label"],
            meta["label_id"],
            meta["session_name"],
            meta["start_idx"],
            meta["end_idx"],
            meta["num_frames"],
            meta["fps"],
            meta["target_bin_mean"],
            meta["motion_mean"],
            meta["motion_std"],
            meta["phase_std"],
            meta["rpm"],
            meta["bpm"],
            meta["created_at"],
        ]
        with open(self.index_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class OfflineReview:
    def __init__(self, cfg: SystemConfig, paths: SessionPaths):
        self.cfg = cfg
        self.paths = paths
        self.recorder = DatasetRecorder(cfg, paths)

    def build_windows_from_session(
        self,
        session_npz_path: str,
        label: str,
        window_sec: float = 2.0,
        stride_sec: float = 0.25,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> List[str]:
        data = np.load(session_npz_path, allow_pickle=True)
        raw = data["raw"]
        timestamps = data["timestamps"]
        frame_count = data["frame_count"]
        motion = data["motion"]
        target_bin = data["target_bin"]
        phase = data["phase"]

        fps = self._estimate_fps(timestamps, self.cfg.fps_est)
        win = max(8, int(window_sec * fps))
        stride = max(1, int(stride_sec * fps))

        s0 = 0 if start_sec is None else max(0, int(start_sec * fps))
        s1 = len(raw) if end_sec is None else min(len(raw), int(end_sec * fps))

        saved = []
        i = s0
        session_name = os.path.basename(os.path.dirname(session_npz_path))
        while i + win <= s1:
            j = i + win
            path = self.recorder.write_segment(
                label=label,
                session_name=session_name,
                raw=raw[i:j],
                timestamps=timestamps[i:j],
                frame_count=frame_count[i:j],
                motion=motion[i:j],
                target_bin=target_bin[i:j],
                phase=phase[i:j],
                start_idx=i,
                end_idx=j,
                fps=fps,
                extra_meta={
                    "source_session": session_npz_path,
                    "window_sec": window_sec,
                    "stride_sec": stride_sec,
                },
            )
            saved.append(path)
            i += stride
        return saved

    @staticmethod
    def _estimate_fps(timestamps: np.ndarray, fallback: float = 28.0) -> float:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.size >= 20:
            dt = np.diff(timestamps)
            dt = dt[dt > 1e-4]
            if dt.size > 0:
                return float(1.0 / np.median(dt))
        return fallback


class FeatureBuilder:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def build_feature_vector(self, npz_path: str) -> Tuple[np.ndarray, int, dict]:
        data = np.load(npz_path, allow_pickle=True)
        motion = np.asarray(data["motion"], dtype=np.float64)
        phase = np.asarray(data["phase"], dtype=np.float64)
        target_bin = np.asarray(data["target_bin"], dtype=np.float64)
        label_id = int(data["label_id"])

        fps = self._fps_from_meta(data, self.cfg.fps_est)
        feats = []

        feats += self._safe_stats(motion)
        mz = SignalTools.robust_z(motion) if motion.size > 0 else np.array([0.0])
        feats += [float(np.max(mz)) if mz.size > 0 else 0.0]
        feats += [float(np.sum(mz > self.cfg.motion_z_cough))]

        if phase.size > 0:
            ph = SignalTools.detrend(phase)
            feats += self._safe_stats(ph)
            resp = SignalTools.bandpass_fft(ph, fps, *self.cfg.resp_band)
            heart = SignalTools.bandpass_fft(ph, fps, *self.cfg.heart_band)
            f_r, p_r = SignalTools.dominant_freq(resp, fps, *self.cfg.resp_band)
            f_h, p_h = SignalTools.dominant_freq(heart, fps, *self.cfg.heart_band)
            feats += [0.0 if f_r is None else float(f_r), float(p_r)]
            feats += [0.0 if f_h is None else float(f_h), float(p_h)]
        else:
            feats += [0.0] * 8

        feats += self._safe_stats(target_bin)

        meta = {}
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"]))
            except Exception:
                meta = {}

        return np.asarray(feats, dtype=np.float32), label_id, meta

    def build_matrix_from_index(self, index_csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        paths = []
        with open(index_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row["segment_path"]
                if os.path.exists(p):
                    paths.append(p)

        X, y = [], []
        for p in paths:
            feat, label_id, _ = self.build_feature_vector(p)
            X.append(feat)
            y.append(label_id)

        if len(X) == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []
        return np.vstack(X), np.asarray(y, dtype=np.int64), paths

    @staticmethod
    def _safe_stats(x: np.ndarray) -> List[float]:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))]

    @staticmethod
    def _fps_from_meta(data, fallback: float = 28.0) -> float:
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"]))
                if "fps" in meta:
                    return float(meta["fps"])
            except Exception:
                pass
        return fallback


class Trainer:
    def __init__(self, model_dir: str = "trained_models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train_random_forest(self, X: np.ndarray, y: np.ndarray, model_name: str = "rf_model"):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, confusion_matrix
            import joblib
        except Exception as e:
            raise RuntimeError("Please install scikit-learn and joblib before training.") from e

        if X.shape[0] < 10:
            raise ValueError("Not enough samples to train. Please record more dataset segments.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)
        model_path = os.path.join(self.model_dir, model_name + ".joblib")
        joblib.dump(clf, model_path)

        report_path = os.path.join(self.model_dir, model_name + "_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\nConfusion Matrix\n")
            f.write(np.array2string(cm))

        print("[TRAIN] model ->", model_path)
        print("[TRAIN] report ->", report_path)
        print(report)
        print(cm)
        return model_path, report_path
