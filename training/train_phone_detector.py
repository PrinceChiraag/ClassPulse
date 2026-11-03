"""Validate and train a one-class phone detector using CPU only."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from validate_phone_dataset import validate


def available_ram_gib() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        if os.name == "nt":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / (1024 ** 3)
        return None


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project / "datasets" / "phone_detection" / "data.yaml")
    parser.add_argument("--base", default="yolo11n.pt", help="Base weights (never modified)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", type=Path, default=project / "models" / "phone_best.pt")
    parser.add_argument("--runs", type=Path, default=project / "runs" / "phone-detection")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        parser.error(f"Dataset YAML not found: {data_yaml}")
    # The user data.yaml is deliberately simple; validate its dataset root via
    # the known project layout so incomplete/invalid data never starts training.
    dataset_root = data_yaml.parent
    errors, report = validate(dataset_root)
    print(f"Dataset validation: {report['totals']}")
    if errors:
        print("Training cancelled because dataset validation failed:")
        for error in errors: print(f" - {error}")
        return 2
    ram = available_ram_gib()
    if ram is not None:
        print(f"Available system memory: {ram:.1f} GiB")
        if ram < 4:
            print("Training cancelled: less than 4 GiB RAM is currently available. Close other applications and retry.")
            return 2
        if ram < 6:
            print("Warning: under 6 GiB RAM is available; batch 2 may still run slowly or exhaust memory.")

    from ultralytics import YOLO
    model = YOLO(args.base)
    result = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                         batch=args.batch, device="cpu", workers=args.workers,
                         patience=args.patience, project=str(args.runs.resolve()),
                         name="classpulse-phone", exist_ok=True)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Ultralytics did not create expected best weights: {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print(f"Best model copied to: {args.output.resolve()}")
    print(f"Training run: {Path(result.save_dir).resolve()}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
