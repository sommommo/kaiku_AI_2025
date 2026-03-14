import os

from config import SystemConfig, SessionPaths
from dataset_tools import FeatureBuilder, Trainer


def main():
    cfg = SystemConfig()
    paths = SessionPaths(cfg)
    paths.ensure()

    index_csv = os.path.join(paths.dataset_root, "index.csv")
    if not os.path.exists(index_csv):
        raise FileNotFoundError(
            f"Cannot find dataset index: {index_csv}\n"
            f"Please record data first, or run offline_review.py to generate windows."
        )

    fb = FeatureBuilder(cfg)
    X, y, sample_paths = fb.build_matrix_from_index(index_csv)

    print("=" * 60)
    print("Training dataset summary")
    print("index_csv:", index_csv)
    print("num_samples:", len(sample_paths))
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("=" * 60)

    if X.shape[0] == 0:
        raise RuntimeError("No valid samples found in index.csv")

    trainer = Trainer(model_dir="trained_models")
    trainer.train_random_forest(X, y, model_name="mmwave_action_rf")


if __name__ == "__main__":
    main()
