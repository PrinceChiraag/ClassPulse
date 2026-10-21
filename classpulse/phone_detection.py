"""Phone object detection for live camera scans.

Uses a COCO-pretrained YOLO model by default. Set CLASSPULSE_PHONE_MODEL to a
fine-tuned weights file to use a custom classroom model.
"""

import os
import threading

import cv2


_model = None
_model_error = None
_model_lock = threading.Lock()


def _get_model():
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model
    with _model_lock:
        if _model is not None or _model_error is not None:
            return _model
        try:
            from ultralytics import YOLO

            weights = os.getenv("CLASSPULSE_PHONE_MODEL", "yolo11n.pt")
            _model = YOLO(weights)
        except Exception as exc:  # optional dependency / model download failure
            _model_error = str(exc)
    return _model


def _phone_class_ids(model):
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    accepted = ("phone", "smartphone", "cell phone", "mobile phone", "handset", "cellular phone")
    return [int(class_id) for class_id, name in names.items() if any(term in str(name).lower() for term in accepted)]


def detect_phones(frame, confidence=0.32):
    """Return room-level phone boxes. Images are evaluated in memory only."""
    model = _get_model()
    if model is None:
        # Shape-based contour matching produced many false positives for books,
        # screens, and classroom furniture. Report unavailable instead of
        # presenting guesses as phone detections.
        return [], "unavailable"

    class_ids = _phone_class_ids(model)
    if not class_ids:
        return [], "no-phone-class"

    try:
        # Ultralytics models are shared by threaded Flask requests; serialize inference.
        with _model_lock:
            result = model.predict(frame, classes=class_ids, conf=float(confidence), imgsz=640, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                detections.append({
                    "label": str(result.names[int(box.cls[0].item())]),
                    "confidence": round(float(box.conf[0].item()), 3),
                    "bbox": {"x": x1, "y": y1, "width": max(1, x2 - x1), "height": max(1, y2 - y1)},
                })
        return detections, "ready"
    except Exception:
        return [], "error"


def _fallback_phone_scan(frame):
    """Find phone-shaped high-contrast rectangles when YOLO weights are absent."""
    if frame is None or not hasattr(frame, "shape"):
        return []
    height, width = frame.shape[:2]
    if width < 80 or height < 80:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 65, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    frame_area = width * height
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        ratio = box_height / max(box_width, 1)
        fill = cv2.contourArea(contour) / max(area, 1)
        if not (frame_area * 0.002 <= area <= frame_area * 0.18 and 1.35 <= ratio <= 2.8 and fill >= 0.25):
            continue
        score = min(0.74, max(0.33, 0.33 + (ratio - 1.35) * 0.12 + fill * 0.12))
        detections.append({
            "label": "phone-shaped object (fallback)",
            "confidence": round(score, 3),
            "bbox": {"x": x, "y": y, "width": box_width, "height": box_height},
        })
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections[:5]
