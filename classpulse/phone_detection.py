"""Phone object detection for live camera scans.

Uses a COCO-pretrained YOLO model by default. Set CLASSPULSE_PHONE_MODEL to a
fine-tuned weights file to use a custom classroom model.
"""

import os
import logging
import threading
import time

import cv2


_model = None
_model_error = None
_retry_after = 0.0
_model_status = None
_model_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _get_model():
    global _model, _model_error, _model_status, _retry_after
    if _model is not None:
        return _model
    if _model_error is not None and time.monotonic() < _retry_after:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        if _model_error is not None and time.monotonic() < _retry_after:
            return None
        try:
            from ultralytics import YOLO

            weights = os.getenv("CLASSPULSE_PHONE_MODEL", "yolo11n.pt")
            _model = YOLO(weights)
            _model_error = None
            _model_status = None
        except Exception as exc:  # optional dependency / model download failure
            _model_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ModuleNotFoundError) and exc.name == "ultralytics":
                _model_status = "not-installed"
            elif isinstance(exc, FileNotFoundError):
                _model_status = "model-not-found"
            else:
                _model_status = "load-error"
            _retry_after = time.monotonic() + 60.0
            logger.exception("Phone detector model failed to load from %s", os.getenv("CLASSPULSE_PHONE_MODEL", "yolo11n.pt"))
    return _model


def _phone_class_ids(model):
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    accepted = ("phone", "smartphone", "cell phone", "mobile phone", "handset", "cellular phone")
    return [int(class_id) for class_id, name in names.items() if any(term in str(name).lower() for term in accepted)]


def phone_detector_runtime_status():
    """Load the configured detector and describe whether it can run here."""
    model = _get_model()
    if model is None:
        return _model_status or "unavailable"
    if _phone_class_ids(model):
        return "ready"
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    return "ready" if str(names.get(67, "")).lower() == "cell phone" else "no-phone-class"


def detect_phones(frame, confidence=0.12):
    """Return room-level phone boxes. Images are evaluated in memory only."""
    model = _get_model()
    if model is None:
        detections = _fallback_phone_scan(frame)
        return detections, "fallback" if detections else (_model_status or "unavailable")

    try:
        class_ids = _phone_class_ids(model)
        if not class_ids:
            # COCO calls this class "cell phone" but not every compatible
            # weights file has that label. Use its canonical COCO class id only
            # when that label set is present; do not fail the entire scan.
            names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
            if str(names.get(67, "")).lower() == "cell phone":
                class_ids = [67]
            else:
                detections = _fallback_phone_scan(frame)
                return detections, "fallback" if detections else "no-phone-class"
        # Ultralytics models are shared by threaded Flask requests; serialize inference.
        with _model_lock:
            # A larger inference scale helps detect smaller phones and partial
            # devices at the edge of a webcam frame. The model still emits boxes
            # clipped to the visible image, so incomplete phones can be flagged.
            result = model.predict(frame, classes=class_ids, conf=float(confidence), imgsz=960, verbose=False)[0]
        detections = _read_result(result)
        return detections, "ready"
    except Exception as exc:
        logger.exception("Phone detection inference failed")
        detections = _fallback_phone_scan(frame)
        return detections, "fallback" if detections else "error"


def _read_result(result, offset_x=0, offset_y=0):
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            detections.append({
                "label": str(result.names[int(box.cls[0].item())]),
                "confidence": round(float(box.conf[0].item()), 3),
                "bbox": {"x": x1 + offset_x, "y": y1 + offset_y,
                         "width": max(1, x2 - x1), "height": max(1, y2 - y1)},
            })
    return detections


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
