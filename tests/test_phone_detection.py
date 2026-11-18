from pathlib import Path
from types import SimpleNamespace

import numpy as np

from classpulse import phone_detection


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class FakeBox:
    def __init__(self, class_id=0, confidence=0.91, xyxy=(10, 20, 50, 80)):
        self.cls = [FakeTensor(class_id)]
        self.conf = [FakeTensor(confidence)]
        self.xyxy = [FakeTensor(list(xyxy))]


def _set_model_state(monkeypatch, *, model=None, status=None, error=None):
    monkeypatch.setattr(phone_detection, "_model", model)
    monkeypatch.setattr(phone_detection, "_model_path", Path("test-model.pt") if model else None)
    monkeypatch.setattr(phone_detection, "_model_status", status)
    monkeypatch.setattr(phone_detection, "_model_error", error)
    monkeypatch.setattr(phone_detection, "_retry_after", 0.0)


def test_phone_class_is_found_by_name_not_fixed_class_id():
    model = SimpleNamespace(names={0: "person", 4: "cell phone", 7: "backpack"})

    assert phone_detection.phone_class_ids(model) == [4]


def test_custom_model_is_the_default_when_no_path_is_configured(monkeypatch):
    phone_model = Path("project/models/phone_best.pt")
    general_model = Path("project/yolo11n.pt")
    monkeypatch.delenv("CLASSPULSE_PHONE_MODEL", raising=False)
    monkeypatch.setattr(phone_detection, "_DEFAULT_PHONE_MODEL", phone_model)
    monkeypatch.setattr(phone_detection, "_GENERAL_MODEL", general_model)
    monkeypatch.setattr(Path, "is_file", lambda path: path.name in {"phone_best.pt", "yolo11n.pt"})

    assert phone_detection._resolve_model_path() == (phone_model, False)


def test_custom_phone_model_load_logs_classes_and_uses_configured_path(monkeypatch):
    model_path = Path("mock-phone.pt")
    fake_model = SimpleNamespace(names={3: "phone"})
    loaded = {}

    def fake_yolo(path, task):
        loaded.update(path=path, task=task)
        return fake_model

    _set_model_state(monkeypatch)
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(phone_detection, "_resolve_model_path", lambda: (model_path, True))
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))

    assert phone_detection._get_model() is fake_model
    assert loaded == {"path": str(model_path), "task": "detect"}
    assert phone_detection.phone_detector_runtime_status() == "ready"


def test_phone_detection_returns_boxes_confidence_and_cpu_settings(monkeypatch):
    fake_model = SimpleNamespace(names={0: "person", 3: "phone"})
    result = SimpleNamespace(names=fake_model.names, boxes=[FakeBox(3, 0.876)])
    captured = {}

    def predict(**kwargs):
        captured.update(kwargs)
        return [result]

    fake_model.predict = predict
    _set_model_state(monkeypatch, model=fake_model)

    detections, status = phone_detection.detect_phones(np.zeros((100, 120, 3), dtype=np.uint8), 0.31)

    assert status == "ready"
    assert detections == [{
        "label": "phone",
        "confidence": 0.876,
        "bbox": {"x": 10, "y": 20, "width": 40, "height": 60},
    }]
    assert captured["classes"] == [3]
    assert captured["device"] == "cpu"
    assert captured["conf"] == 0.31


def test_person_class_is_never_used_as_phone_class(monkeypatch):
    fake_model = SimpleNamespace(names={0: "person", 2: "backpack"})
    _set_model_state(monkeypatch, model=fake_model)

    detections, status = phone_detection.detect_phones(np.zeros((80, 80, 3), dtype=np.uint8))

    assert detections == []
    assert status == "no-phone-class"


def test_no_phone_frame_returns_valid_empty_detections(monkeypatch):
    fake_model = SimpleNamespace(names={0: "phone"})
    fake_model.predict = lambda **_kwargs: [SimpleNamespace(names=fake_model.names, boxes=None)]
    _set_model_state(monkeypatch, model=fake_model)

    detections, status = phone_detection.detect_phones(np.zeros((80, 80, 3), dtype=np.uint8))

    assert detections == []
    assert status == "ready"


def test_invalid_frame_is_rejected_without_inference(monkeypatch):
    fake_model = SimpleNamespace(names={0: "phone"})
    fake_model.predict = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not infer"))
    _set_model_state(monkeypatch, model=fake_model)

    detections, status = phone_detection.detect_phones(np.zeros((80, 80), dtype=np.uint8))

    assert detections == []
    assert status == "invalid-frame"


def test_model_loading_failure_is_reported_without_fake_fallback(monkeypatch):
    _set_model_state(monkeypatch)
    missing = Path("missing.pt")
    monkeypatch.setattr(phone_detection, "_resolve_model_path", lambda: (missing, True))

    detections, status = phone_detection.detect_phones(np.zeros((80, 80, 3), dtype=np.uint8))

    assert detections == []
    assert status == "model-not-found"


def test_object_scan_combines_custom_phone_and_general_coco_objects(monkeypatch):
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    phone = {
        "label": "phone",
        "confidence": 0.88,
        "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
    }
    general = SimpleNamespace(
        names={0: "person", 67: "cell phone"},
        predict=lambda **kwargs: [SimpleNamespace(
            names={0: "person", 67: "cell phone"},
            boxes=[FakeBox(0, 0.91), FakeBox(67, 0.82)],
        )],
    )
    monkeypatch.setattr(phone_detection, "detect_phones", lambda *_args, **_kwargs: ([phone], "ready"))
    monkeypatch.setattr(phone_detection, "_get_general_model", lambda: general)

    detections, status = phone_detection.detect_objects(frame)

    assert status == "ready"
    assert [item["label"] for item in detections] == ["phone", "person"]
    assert detections[0] == phone


def test_general_model_phone_detection_is_used_if_custom_phone_model_fails(monkeypatch):
    general = SimpleNamespace(
        predict=lambda **_kwargs: [SimpleNamespace(
            names={67: "cell phone", 73: "book"},
            boxes=[FakeBox(67, 0.77), FakeBox(73, 0.81)],
        )],
    )
    monkeypatch.setattr(phone_detection, "detect_phones", lambda *_args, **_kwargs: ([], "load-error"))
    monkeypatch.setattr(phone_detection, "_get_general_model", lambda: general)

    detections, status = phone_detection.detect_objects(np.zeros((80, 80, 3), dtype=np.uint8))

    assert status == "partial"
    assert [item["label"] for item in detections] == ["cell phone", "book"]
