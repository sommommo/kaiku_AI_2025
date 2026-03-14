import argparse
import json
import os
import time

import numpy as np

from config import SystemConfig, SessionPaths
from dataset_tools import FeatureBuilder, ID_TO_LABEL


def load_model(model_path: str):
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("Please install joblib before running predict_model.py") from e
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_segment(model, cfg: SystemConfig, npz_path: str):
    fb = FeatureBuilder(cfg)
    feat, true_label_id, meta = fb.build_feature_vector(npz_path)
    pred_id = int(model.predict(feat.reshape(1, -1))[0])

    prob = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(feat.reshape(1, -1))[0]
        prob = {ID_TO_LABEL.get(i, str(i)): float(v) for i, v in enumerate(p)}

    result = {
        "segment_path": npz_path,
        "pred_label_id": pred_id,
        "pred_label": ID_TO_LABEL.get(pred_id, str(pred_id)),
        "true_label_id": true_label_id,
        "true_label": ID_TO_LABEL.get(true_label_id, str(true_label_id)),
        "prob": prob,
        "meta": meta,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict label for one saved mmWave segment.")
    parser.add_argument("--segment", type=str, required=True, help="Path to one segment .npz")
    parser.add_argument("--model", type=str, default=os.path.join("trained_models", "mmwave_action_rf.joblib"))
    args = parser.parse_args()

    cfg = SystemConfig()
    paths = SessionPaths(cfg)
    paths.ensure()

    model = load_model(args.model)
    result = predict_segment(model, cfg, args.segment)

    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
