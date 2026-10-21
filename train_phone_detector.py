"""Fine-tune a phone detector for classroom camera images.

Dataset format: Ultralytics YOLO detection format, one class named `phone`.
See datasets/phone/data.example.yaml and README.md for the expected folders.
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a YOLO dataset YAML file")
    parser.add_argument("--base", default="yolo11n.pt", help="Pretrained base weights")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. 0 or cpu")
    args = parser.parse_args()

    model = YOLO(args.base)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/phone-detection",
        name="classpulse-phone",
        exist_ok=True,
    )
    save_dir = getattr(result, "save_dir", "runs/phone-detection/classpulse-phone")
    print(f"Training complete. Set CLASSPULSE_PHONE_MODEL to {save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
