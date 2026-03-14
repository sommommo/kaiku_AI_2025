import os
import pickle
import numpy as np
from typing import Optional, Dict, Any


class MotionModelRunner:
    def __init__(self, model_path: Optional[str], label_map: Dict[int, str], score_threshold: float = 0.5):
        self.model_path = model_path
        self.label_map = label_map
        self.score_threshold = score_threshold
        self.model = None
        if model_path:
            self.load(model_path)

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        try:
            import joblib
            self.model = joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        self.model_path = model_path

    def build_feature_vector(self, phase_wave: np.ndarray, motion_wave: np.ndarray, range_profile: np.ndarray,
                             fps: float, rpm, bpm, target_bin: int) -> np.ndarray:
        phase_wave = np.asarray(phase_wave, dtype=np.float64)
        motion_wave = np.asarray(motion_wave, dtype=np.float64)
        range_profile = np.asarray(range_profile, dtype=np.float64)

        def _safe(v, d=0.0):
            return d if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)

        feats = [
            float(fps),
            _safe(target_bin, -1),
            _safe(rpm),
            _safe(bpm),
            float(np.mean(phase_wave)) if phase_wave.size else 0.0,
            float(np.std(phase_wave)) if phase_wave.size else 0.0,
            float(np.max(np.abs(phase_wave))) if phase_wave.size else 0.0,
            float(np.mean(motion_wave)) if motion_wave.size else 0.0,
            float(np.std(motion_wave)) if motion_wave.size else 0.0,
            float(np.max(motion_wave)) if motion_wave.size else 0.0,
            float(np.mean(range_profile)) if range_profile.size else 0.0,
            float(np.std(range_profile)) if range_profile.size else 0.0,
            float(np.max(range_profile)) if range_profile.size else 0.0,
        ]
        return np.asarray(feats, dtype=np.float32)

    def predict(self, feat: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            return {"label": "-", "score": 0.0, "prob": None}

        pred_id = int(self.model.predict(feat.reshape(1, -1))[0])
        label = self.label_map.get(pred_id, str(pred_id))
        score = 1.0
        prob = None

        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(feat.reshape(1, -1))[0]
            best = int(np.argmax(p))
            pred_id = best
            label = self.label_map.get(best, str(best))
            score = float(p[best])
            prob = {self.label_map.get(i, str(i)): float(v) for i, v in enumerate(p)}

        return {
            "label": label,
            "score": float(score),
            "prob": prob,
            "is_confident": bool(score >= self.score_threshold),
        }
