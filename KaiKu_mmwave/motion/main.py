import logging

from config import SystemConfig, SessionPaths, KEY_LABELS
from gui import LiveWindow
from kkt_runner import KKTRunner
from system_core import RadarSystem


def main():
    logging.disable(logging.CRITICAL)

    cfg = SystemConfig()
    paths = SessionPaths(cfg)
    paths.ensure()

    system = RadarSystem(cfg, paths)
    window = LiveWindow(max_gui_points=cfg.max_gui_points)
    runner = KKTRunner(cfg, system, window)

    print("=" * 60)
    print("mmWave Research System (Modular)")
    print("Keys for manual labeling (future extension):")
    for k, v in KEY_LABELS.items():
        print(f"  {k} -> {v}")
    print("Press Ctrl+C to stop and save session.")
    print("=" * 60)

    try:
        runner.start()
        runner.loop()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt -> stopping...")
    finally:
        system.save_session()
        runner.stop()
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
