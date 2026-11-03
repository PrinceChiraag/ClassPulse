"""Evaluate phone detector precision, recall and mAP on the held-out test split."""
import argparse
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=project / "models" / "phone_best.pt")
    parser.add_argument("--data", type=Path, default=project / "datasets" / "phone_detection" / "data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if not args.model.is_file(): parser.error(f"Model not found: {args.model}")
    if not args.data.is_file(): parser.error(f"Dataset YAML not found: {args.data}")
    from ultralytics import YOLO
    metrics = YOLO(str(args.model)).val(data=str(args.data.resolve()), split="test", imgsz=args.imgsz, batch=args.batch, device="cpu")
    box = metrics.box
    print(f"Precision: {float(box.mp):.4f}")
    print(f"Recall: {float(box.mr):.4f}")
    print(f"mAP50: {float(box.map50):.4f}")
    print(f"mAP50-95: {float(box.map):.4f}")


if __name__ == "__main__": main()
