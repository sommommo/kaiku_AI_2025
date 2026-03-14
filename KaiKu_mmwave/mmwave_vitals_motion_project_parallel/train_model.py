import os
from dataset_tools import FeatureBuilder, Trainer
from config import DATASET_ROOT


def main():
    index_csv = os.path.join(DATASET_ROOT, "index.csv")
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Cannot find dataset index: {index_csv}")
    fb = FeatureBuilder()
    X, y, _ = fb.build_matrix_from_index(index_csv)
    if X.shape[0] == 0:
        raise RuntimeError("No samples found in dataset index")
    trainer = Trainer()
    out = trainer.train_random_forest(X, y, model_name="mmwave_action_rf")
    print("Saved model ->", out)


if __name__ == "__main__":
    main()
