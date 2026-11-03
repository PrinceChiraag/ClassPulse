"""Run the trained detector on an image or video and save annotated outputs."""
import argparse
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Image or video file")
    parser.add_argument("--model", type=Path, default=project / "models" / "phone_best.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=project / "runs" / "phone-predictions")
    args = parser.parse_args()
    if not args.source.is_file(): parser.error(f"Input file not found: {args.source}")
    if not args.model.is_file(): parser.error(f"Model not found: {args.model}")
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    results = model.predict(source=str(args.source), conf=args.confidence, imgsz=args.imgsz,
                            device="cpu", save=True, project=str(args.output.parent),
                            name=args.output.name, exist_ok=True, verbose=False)
    for result in results:
        for box in result.boxes if result.boxes is not None else []:
            print(f"{result.path}: {result.names[int(box.cls[0])]} confidence={float(box.conf[0]):.3f}")
    print(f"Annotated outputs saved under {args.output.resolve()}")


if __name__ == "__main__": main()
