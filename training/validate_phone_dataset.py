"""Validate the ClassPulse one-class YOLO phone dataset before training."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import cv2

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate(root: Path, minimum_positive: int = 1, minimum_negative: int = 1) -> tuple[list[str], dict]:
    errors: list[str] = []
    report = {"splits": {}, "totals": {"images": 0, "positive": 0, "negative": 0, "boxes": 0}}
    hashes: dict[str, tuple[str, Path]] = {}
    for split in SPLITS:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        if not image_dir.is_dir(): errors.append(f"Missing folder: {image_dir}")
        if not label_dir.is_dir(): errors.append(f"Missing folder: {label_dir}")
        if not image_dir.is_dir() or not label_dir.is_dir(): continue
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        image_stems = {p.stem for p in images}
        if len(image_stems) != len(images): errors.append(f"{split}: duplicate image stems with different extensions")
        positive = negative = boxes = 0
        if not images: errors.append(f"{split}: no supported images")
        for image in images:
            if cv2.imread(str(image)) is None:
                errors.append(f"Unreadable image: {image}")
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                previous_split, previous_path = hashes[digest]
                errors.append(f"Duplicate image content in {previous_split} and {split}: {previous_path} / {image}")
            else:
                hashes[digest] = (split, image)
            label = labels.get(image.stem)
            if label is None:
                errors.append(f"Missing label file for {image}")
                continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if not lines:
                negative += 1
                continue
            positive += 1
            for line_no, line in enumerate(lines, 1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{label}:{line_no}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0]); coords = [float(v) for v in fields[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_no}: invalid numeric annotation")
                    continue
                if class_id != 0: errors.append(f"{label}:{line_no}: class id must be 0")
                if not all(0 <= value <= 1 for value in coords): errors.append(f"{label}:{line_no}: coordinates must be in [0,1]")
                if coords[2] <= 0 or coords[3] <= 0: errors.append(f"{label}:{line_no}: box width and height must be positive")
                else: boxes += 1
        for stem in sorted(labels.keys() - image_stems): errors.append(f"Orphan label without image: {labels[stem]}")
        report["splits"][split] = {"images": len(images), "positive": positive, "negative": negative, "boxes": boxes}
        for key, value in report["splits"][split].items(): report["totals"][key] += value
    if report["totals"]["positive"] < minimum_positive: errors.append(f"Need at least {minimum_positive} positive images; found {report['totals']['positive']}")
    if report["totals"]["negative"] < minimum_negative: errors.append(f"Need at least {minimum_negative} negative images; found {report['totals']['negative']}")
    return errors, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "datasets" / "phone_detection")
    parser.add_argument("--minimum-positive", type=int, default=1)
    parser.add_argument("--minimum-negative", type=int, default=1)
    args = parser.parse_args()
    errors, report = validate(args.root, args.minimum_positive, args.minimum_negative)
    print(f"Dataset: {args.root.resolve()}")
    for split, values in report["splits"].items(): print(f"{split}: {values['images']} images; {values['positive']} positive; {values['negative']} negative; {values['boxes']} boxes")
    print(f"Total: {report['totals']}")
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} issue(s))")
        for error in errors: print(f" - {error}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__": raise SystemExit(main())
