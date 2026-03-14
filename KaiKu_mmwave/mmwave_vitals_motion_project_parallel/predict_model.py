import argparse
from dataset_tools import FeatureBuilder, ID_TO_LABEL
from model_runner import MotionModelRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    fb = FeatureBuilder()
    feat, true_id, meta = fb.build_feature_vector(args.segment)
    runner = MotionModelRunner(args.model, ID_TO_LABEL)
    pred = runner.predict(feat)
    print({
        "segment": args.segment,
        "pred": pred,
        "true": ID_TO_LABEL.get(true_id, str(true_id)),
        "meta": meta,
    })


if __name__ == "__main__":
    main()
