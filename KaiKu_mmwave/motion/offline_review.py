import argparse
import os

from config import SystemConfig, SessionPaths
from dataset_tools import OfflineReview


def main():
    parser = argparse.ArgumentParser(description="Build sliding-window dataset segments from a saved session.")
    parser.add_argument("--session_npz", type=str, required=True, help="Path to saved raw_frames.npz")
    parser.add_argument("--label", type=str, required=True, help="Label for all exported windows, e.g. fall/cough/sit/normal")
    parser.add_argument("--window_sec", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--stride_sec", type=float, default=0.25, help="Stride length in seconds")
    parser.add_argument("--start_sec", type=float, default=None, help="Optional start time in seconds")
    parser.add_argument("--end_sec", type=float, default=None, help="Optional end time in seconds")
    args = parser.parse_args()

    if not os.path.exists(args.session_npz):
        raise FileNotFoundError(f"Session file not found: {args.session_npz}")

    cfg = SystemConfig()
    paths = SessionPaths(cfg)
    paths.ensure()

    review = OfflineReview(cfg, paths)
    saved = review.build_windows_from_session(
        session_npz_path=args.session_npz,
        label=args.label,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

    print("=" * 60)
    print("Offline review finished")
    print("session_npz:", args.session_npz)
    print("label:", args.label)
    print("num_windows:", len(saved))
    print("dataset_dir:", paths.dataset_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
