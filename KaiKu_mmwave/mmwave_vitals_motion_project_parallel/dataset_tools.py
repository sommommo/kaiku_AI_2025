import os
import csv
import time
import numpy as np

from config import DATASET_ROOT, TRAINED_MODEL_DIR
from dsp import bandpass_fft, dominant_freq, detrend_linear

LABEL_TO_ID = {
    "normal": 0,
    "fall": 1,
    "cough": 2,
    "agitation": 3,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class FeatureBuilder:
    def build_feature_vector(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        phase = np.asarray(data["phase_wave"], dtype=np.float64)
        motion = np.asarray(data["motion_wave"], dtype=np.float64)
        range_profile = np.asarray(data["range_profile"], dtype=np.float64)
        fps = float(data["fps"])
        target_bin = int(data["target_bin"])
        label_id = int(data["label_id"])

        ph = detrend_linear(phase) if phase.size else phase
        resp = bandpass_fft(ph, fps, 0.10, 0.60) if phase.size else phase
        heart = bandpass_fft(ph, fps, 0.80, 3.00) if phase.size else phase
        fr, _ = dominant_freq(resp, fps, 0.10, 0.60) if phase.size else (None, 0.0)
        fh, _ = dominant_freq(heart, fps, 0.80, 3.00) if phase.size else (None, 0.0)
        rpm = None if fr is None else fr * 60.0
        bpm = None if fh is None else fh * 60.0

        feat = np.asarray([
            fps, target_bin,
            0.0 if rpm is None else rpm,
            0.0 if bpm is None else bpm,
            float(np.mean(ph)) if ph.size else 0.0,
            float(np.std(ph)) if ph.size else 0.0,
            float(np.max(np.abs(ph))) if ph.size else 0.0,
            float(np.mean(motion)) if motion.size else 0.0,
            float(np.std(motion)) if motion.size else 0.0,
            float(np.max(motion)) if motion.size else 0.0,
            float(np.mean(range_profile)) if range_profile.size else 0.0,
            float(np.std(range_profile)) if range_profile.size else 0.0,
            float(np.max(range_profile)) if range_profile.size else 0.0,
        ], dtype=np.float32)
        return feat, label_id, {"rpm": rpm, "bpm": bpm, "target_bin": target_bin}

    def build_matrix_from_index(self, index_csv: str):
        X, y, paths = [], [], []
        with open(index_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row["segment_path"]
                if not os.path.exists(p):
                    continue
                feat, label_id, _ = self.build_feature_vector(p)
                X.append(feat)
                y.append(label_id)
                paths.append(p)
        if not X:
            return np.empty((0, 13), dtype=np.float32), np.empty((0,), dtype=np.int64), []
        return np.vstack(X), np.asarray(y, dtype=np.int64), paths


class DatasetRecorder:
    def __init__(self, dataset_root: str = DATASET_ROOT):
        self.dataset_root = dataset_root
        self.segment_root = os.path.join(dataset_root, "segments")
        self.index_csv = os.path.join(dataset_root, "index.csv")
        os.makedirs(self.segment_root, exist_ok=True)
        if not os.path.exists(self.index_csv):
            with open(self.index_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["segment_path", "label", "label_id", "fps", "target_bin", "created_at"])

    def save_segment(self, label: str, phase_wave, motion_wave, range_profile, fps: float, target_bin: int):
        label = label if label in LABEL_TO_ID else "normal"
        out_dir = os.path.join(self.segment_root, label)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, time.strftime(f"{label}_%Y%m%d_%H%M%S.npz"))
        np.savez_compressed(
            path,
            label=label,
            label_id=LABEL_TO_ID[label],
            fps=float(fps),
            target_bin=int(target_bin),
            phase_wave=np.asarray(phase_wave),
            motion_wave=np.asarray(motion_wave),
            range_profile=np.asarray(range_profile),
        )
        with open(self.index_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([path, label, LABEL_TO_ID[label], fps, target_bin, time.strftime("%Y-%m-%d %H:%M:%S")])
        return path


class Trainer:
    def __init__(self, model_dir: str = TRAINED_MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train_random_forest(self, X, y, model_name: str = "mmwave_action_rf"):
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        model.fit(X, y)
        out = os.path.join(self.model_dir, model_name + ".joblib")
        joblib.dump(model, out)
        return out
