"""Lazy, CPU-only YOLO inference for captured ClassPulse frames."""

import logging
import math
import os
import threading
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PHONE_MODEL = _PROJECT_ROOT / "models" / "phone_best.pt"
_GENERAL_MODEL = _PROJECT_ROOT / "yolo11n.pt"
_PHONE_LABELS = {
    "phone",
    "smartphone",
    "cell phone",
    "mobile phone",
    "handset",
    "cellular phone",
}

_model = None
_model_path = None
_model_error = None
_model_status = None
_retry_after = 0.0
_general_model = None
_general_model_error = None
_general_model_status = None
_general_retry_after = 0.0
_model_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _resolve_model_path():
    configured = os.getenv("CLASSPULSE_PHONE_MODEL", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path.resolve(), True

    if _DEFAULT_PHONE_MODEL.is_file():
        return _DEFAULT_PHONE_MODEL, False
    return _GENERAL_MODEL, False


def _normalize_names(names):
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    return {int(class_id): str(label).strip() for class_id, label in (names or {}).items()}


def phone_class_ids(model):
    """Find phone classes by label; never infer a class from its numeric ID."""
    return [
        class_id
        for class_id, label in _normalize_names(getattr(model, "names", {})).items()
        if is_phone_label(label)
    ]


def is_phone_label(label):
    return str(label or "").strip().casefold() in _PHONE_LABELS


def _get_model():
    """Load one selected model per process and report load failures explicitly."""
    global _model, _model_path, _model_error, _model_status, _retry_after
    if _model is not None:
        return _model
    if _model_error is not None and time.monotonic() < _retry_after:
        return None

    with _model_lock:
        if _model is not None:
            return _model
        if _model_error is not None and time.monotonic() < _retry_after:
            return None

        path, explicit = _resolve_model_path()
        logger.info("Loading YOLO detector from %s (exists=%s)", path, path.is_file())
        if not path.is_file():
            _model_status = "model-not-found"
            _model_error = f"Model file does not exist: {path}"
            _retry_after = time.monotonic() + 60.0
            if explicit:
                logger.error("Configured phone model is missing: %s", path)
                return None
            if path != _GENERAL_MODEL and _GENERAL_MODEL.is_file():
                logger.warning("Default phone model is missing; falling back to COCO model %s", _GENERAL_MODEL)
                path = _GENERAL_MODEL
            else:
                logger.error("No YOLO model file is available at %s", path)
                return None

        try:
            from ultralytics import YOLO

            loaded_model = YOLO(str(path), task="detect")
            names = _normalize_names(getattr(loaded_model, "names", {}))
            if not names:
                raise RuntimeError(f"YOLO model at {path} has no class names")

            _model = loaded_model
            _model_path = path
            _model_error = None
            _model_status = "ready"
            _retry_after = 0.0
            logger.info(
                "YOLO detector loaded on demand from %s; classes=%s; phone_class_ids=%s",
                path,
                names,
                phone_class_ids(loaded_model),
            )
            return _model
        except Exception as exc:
            _model = None
            _model_path = path
            _model_error = f"{type(exc).__name__}: {exc}"
            _model_status = (
                "not-installed"
                if isinstance(exc, ModuleNotFoundError) and exc.name == "ultralytics"
                else "model-not-found"
                if isinstance(exc, FileNotFoundError)
                else "load-error"
            )
            _retry_after = time.monotonic() + 60.0
            logger.exception("YOLO model loading failed for %s", path)
            return None


def phone_detector_runtime_status():
    """Return detector status without forcing heavyweight model loading."""
    if _model is None:
        return _model_status or "not-loaded"
    return "ready" if phone_class_ids(_model) else "no-phone-class"


def object_detector_runtime_status():
    """Report general object detector state without forcing model loading."""
    if _general_model is not None or (_model is not None and _model_path == _GENERAL_MODEL):
        return "ready"
    return _general_model_status or "not-loaded"


def _get_general_model():
    """Load the bundled COCO detector once for non-phone object classes."""
    global _general_model, _general_model_error, _general_model_status, _general_retry_after
    if _model is not None and _model_path == _GENERAL_MODEL:
        return _model
    if _general_model is not None:
        return _general_model
    if _general_model_error is not None and time.monotonic() < _general_retry_after:
        return None

    with _model_lock:
        if _model is not None and _model_path == _GENERAL_MODEL:
            return _model
        if _general_model is not None:
            return _general_model
        if _general_model_error is not None and time.monotonic() < _general_retry_after:
            return None
        logger.info("Loading general YOLO object detector from %s (exists=%s)", _GENERAL_MODEL, _GENERAL_MODEL.is_file())
        if not _GENERAL_MODEL.is_file():
            _general_model_status = "model-not-found"
            _general_model_error = f"Model file does not exist: {_GENERAL_MODEL}"
            _general_retry_after = time.monotonic() + 60.0
            logger.error("General YOLO model is missing: %s", _GENERAL_MODEL)
            return None
        try:
            from ultralytics import YOLO

            model = YOLO(str(_GENERAL_MODEL), task="detect")
            names = _normalize_names(getattr(model, "names", {}))
            if not names:
                raise RuntimeError(f"YOLO model at {_GENERAL_MODEL} has no class names")
            _general_model = model
            _general_model_error = None
            _general_model_status = "ready"
            _general_retry_after = 0.0
            logger.info("General YOLO detector loaded from %s; classes=%s", _GENERAL_MODEL, names)
            return _general_model
        except Exception as exc:
            _general_model = None
            _general_model_error = f"{type(exc).__name__}: {exc}"
            _general_model_status = "not-installed" if isinstance(exc, ModuleNotFoundError) and exc.name == "ultralytics" else "load-error"
            _general_retry_after = time.monotonic() + 60.0
            logger.exception("General YOLO model loading failed for %s", _GENERAL_MODEL)
            return None


def _valid_frame(frame):
    return (
        frame is not None
        and getattr(frame, "ndim", 0) == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
        and frame.shape[2] == 3
    )


def _normalize_confidence(confidence):
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.25
    if not math.isfinite(value):
        value = 0.25
    return max(0.10, min(0.8, value))


def detect_objects(frame, confidence=0.25):
    """Combine the tuned phone detector with general COCO object detections."""
    if not _valid_frame(frame):
        logger.warning("Object inference rejected an invalid frame")
        return [], "invalid-frame"
    threshold = _normalize_confidence(confidence)
    height, width = frame.shape[:2]
    started = time.monotonic()
    phone_detections, phone_status = detect_phones(frame, confidence=threshold)
    general_model = _get_general_model()
    general_detections = []
    general_inference_status = _general_model_status or "unavailable"

    if general_model is not None:
        try:
            with _model_lock:
                result = general_model.predict(
                    source=frame,
                    conf=threshold,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]
            general_detections = _read_result(result)
            general_inference_status = "ready"
            logger.info(
                "General YOLO inference completed: model=%s frame=%sx%s detections=%s duration_ms=%s",
                _GENERAL_MODEL,
                width,
                height,
                len(general_detections),
                round((time.monotonic() - started) * 1000),
            )
        except Exception:
            general_inference_status = "error"
            logger.exception(
                "General YOLO inference failed: model=%s frame=%sx%s confidence=%.3f",
                _GENERAL_MODEL,
                width,
                height,
                threshold,
            )

    # The custom phone model is more specific. Use COCO phone labels only if
    # that model is unavailable; all other COCO classes remain visible.
    general_phones = [item for item in general_detections if is_phone_label(item["label"])]
    other_objects = [item for item in general_detections if not is_phone_label(item["label"])]
    if phone_status == "ready":
        detections = phone_detections + other_objects
    else:
        detections = general_detections

    if phone_status == "ready" and general_inference_status == "ready":
        status = "ready"
    elif phone_status == "ready" or general_inference_status == "ready":
        status = "partial"
    else:
        status = phone_status if phone_status not in {"not-loaded", "unavailable"} else general_inference_status

    logger.info(
        "YOLO combined scan completed: frame=%sx%s objects=%s phones=%s phone_model=%s general_model=%s",
        width,
        height,
        len(detections),
        len(phone_detections) if phone_status == "ready" else len(general_phones),
        phone_status,
        general_inference_status,
    )
    return detections, status


def detect_phones(frame, confidence=0.25):
    """Return detections only for class labels that explicitly identify phones."""
    if not _valid_frame(frame):
        logger.warning("Phone inference rejected an invalid frame")
        return [], "invalid-frame"

    model = _get_model()
    if model is None:
        return [], _model_status or "unavailable"

    class_ids = phone_class_ids(model)
    if not class_ids:
        logger.error(
            "Loaded model %s has no recognized phone class; classes=%s",
            _model_path,
            _normalize_names(getattr(model, "names", {})),
        )
        return [], "no-phone-class"

    threshold = _normalize_confidence(confidence)
    height, width = frame.shape[:2]
    started = time.monotonic()
    try:
        with _model_lock:
            result = model.predict(
                source=frame,
                classes=class_ids,
                conf=threshold,
                imgsz=640,
                device="cpu",
                verbose=False,
            )[0]
        detections = _read_result(result)
        logger.info(
            "YOLO phone inference completed: model=%s class_ids=%s frame=%sx%s detections=%s confidences=%s duration_ms=%s",
            _model_path,
            class_ids,
            width,
            height,
            len(detections),
            [item["confidence"] for item in detections],
            round((time.monotonic() - started) * 1000),
        )
        return detections, "ready"
    except Exception:
        logger.exception(
            "YOLO phone inference failed: model=%s class_ids=%s frame=%sx%s confidence=%.3f",
            _model_path,
            class_ids,
            width,
            height,
            threshold,
        )
        return [], "error"


def _read_result(result, offset_x=0, offset_y=0):
    detections = []
    names = _normalize_names(getattr(result, "names", {}))
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        label = names.get(class_id)
        if label is None:
            logger.warning("YOLO returned unknown class id %s; skipping box", class_id)
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": label,
                "confidence": round(float(box.conf[0].item()), 3),
                "bbox": {
                    "x": x1 + offset_x,
                    "y": y1 + offset_y,
                    "width": max(1, x2 - x1),
                    "height": max(1, y2 - y1),
                },
            }
        )
    return detections
