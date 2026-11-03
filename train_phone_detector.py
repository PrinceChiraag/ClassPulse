"""Backward-compatible launcher for the validated CPU phone trainer."""
from pathlib import Path
import sys


if __name__ == "__main__":
    training_dir = Path(__file__).resolve().parent / "training"
    sys.path.insert(0, str(training_dir))
    from train_phone_detector import main

    raise SystemExit(main())
