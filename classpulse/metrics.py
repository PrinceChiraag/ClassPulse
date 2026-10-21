from collections import deque
from typing import Optional

# MediaPipe face mesh indices for eye landmarks.
LEFT_EYE = {
    "outer": 33,
    "inner": 133,
    "upper": 159,
    "lower": 145,
}
RIGHT_EYE = {
    "outer": 263,
    "inner": 362,
    "upper": 386,
    "lower": 374,
}

MOUTH_TOP = 13
MOUTH_BOTTOM = 14

EAR_WINDOW = deque(maxlen=60)
MAR_WINDOW = deque(maxlen=60)
PITCH_WINDOW = deque(maxlen=60)

DEFAULT_DETECTION_THRESHOLDS = {
    "ear_blink_cutoff": 0.18,
    "mar_yawn_sensitivity": 0.24,
    "posture_tilt_degrees": 18.0,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_thresholds(thresholds: Optional[dict] = None) -> dict:
    settings = dict(DEFAULT_DETECTION_THRESHOLDS)
    if thresholds:
        for key, value in thresholds.items():
            if key in settings and value is not None:
                settings[key] = float(value)
    return settings


def _normalized_distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _rolling_fatigue_state(
    ear_value: float,
    mar_value: float,
    pitch_value: float,
    thresholds: Optional[dict] = None,
    rolling_state: Optional[dict] = None,
) -> dict:
    settings = _normalize_thresholds(thresholds)
    state = rolling_state or {"ear": EAR_WINDOW, "mar": MAR_WINDOW, "pitch": PITCH_WINDOW}
    state["ear"].append(float(ear_value))
    state["mar"].append(float(mar_value))
    state["pitch"].append(float(pitch_value))

    ear_samples = list(state["ear"])
    mar_samples = list(state["mar"])
    pitch_samples = list(state["pitch"])
    ear_threshold = float(settings["ear_blink_cutoff"])
    mar_threshold = float(settings["mar_yawn_sensitivity"])
    slouch_threshold = float(settings["posture_tilt_degrees"]) / 100.0

    low_ear_count = sum(1 for sample in ear_samples if sample < ear_threshold)
    high_mar_count = sum(1 for sample in mar_samples if sample > mar_threshold)
    slouching_count = sum(1 for sample in pitch_samples if sample > slouch_threshold)

    # The browser sends sampled frames rather than a full video stream. Require a
    # sustained closure pattern over at least 12 samples to avoid treating blinks
    # as sleep while still allowing live sampling to raise an alert promptly.
    microsleep = len(ear_samples) >= 12 and low_ear_count >= max(12, int(len(ear_samples) * 0.65))
    yawning = high_mar_count >= 75
    slouching = slouching_count >= 30
    drowsiness_state = "Alert"
    if yawning:
        drowsiness_state = "Yawning"
    elif microsleep or slouching:
        drowsiness_state = "Drowsy"

    low_ear_ratio = low_ear_count / max(len(ear_samples), 1)
    high_mar_ratio = high_mar_count / max(len(mar_samples), 1)
    slouching_ratio = slouching_count / max(len(pitch_samples), 1)
    fatigue_index = (low_ear_ratio * 45.0) + (high_mar_ratio * 35.0) + (slouching_ratio * 20.0)
    fatigue_index = round(_clamp(fatigue_index, 0.0, 100.0), 1)

    return {
        "microsleep": bool(microsleep),
        "yawning": bool(yawning),
        "drowsiness_state": drowsiness_state,
        "fatigue_index": fatigue_index,
    }


def compute_attention_fatigue(face_landmarks, pose_landmarks, thresholds: Optional[dict] = None, rolling_state: Optional[dict] = None) -> dict:
    settings = _normalize_thresholds(thresholds)
    if face_landmarks is None:
        return {
            "attention": 0.0,
            "fatigue": 1.0,
            "eye_engagement": 0.0,
            "posture_score": 0.0,
            "microsleep": False,
            "yawning": False,
            "drowsiness_state": "Alert",
            "fatigue_index": 100.0,
            "detection_thresholds": settings,
        }

    left_upper = face_landmarks[LEFT_EYE["upper"]]
    left_lower = face_landmarks[LEFT_EYE["lower"]]
    left_outer = face_landmarks[LEFT_EYE["outer"]]
    left_inner = face_landmarks[LEFT_EYE["inner"]]

    right_upper = face_landmarks[RIGHT_EYE["upper"]]
    right_lower = face_landmarks[RIGHT_EYE["lower"]]
    right_outer = face_landmarks[RIGHT_EYE["outer"]]
    right_inner = face_landmarks[RIGHT_EYE["inner"]]

    upper_lip = face_landmarks[MOUTH_TOP]
    lower_lip = face_landmarks[MOUTH_BOTTOM]

    left_ear = _normalized_distance(left_upper, left_lower) / max(
        _normalized_distance(left_outer, left_inner), 1e-6
    )
    right_ear = _normalized_distance(right_upper, right_lower) / max(
        _normalized_distance(right_outer, right_inner), 1e-6
    )
    average_ear = (left_ear + right_ear) / 2.0
    mar = _normalized_distance(upper_lip, lower_lip) / max(
        _normalized_distance(face_landmarks[61], face_landmarks[291]), 1e-6
    )

    if pose_landmarks is not None:
        nose = pose_landmarks.landmark[0]
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        shoulders_midpoint = (left_shoulder.y + right_shoulder.y) / 2.0
        pitch_tilt = abs(nose.y - shoulders_midpoint)
        posture_tilt_threshold = float(settings["posture_tilt_degrees"]) / 100.0
        posture_score = _clamp(1.0 - max(0.0, pitch_tilt - posture_tilt_threshold) * 4.0, 0.0, 1.0)
    else:
        posture_score = 0.7
        pitch_tilt = 0.12

    fatigue = 0.25 + max(0.0, 0.35 - average_ear) * 1.2 + (1.0 - posture_score) * 0.3
    attention = max(0.0, min(100.0, 100.0 * (average_ear * 1.75 + posture_score * 0.25)))

    if attention < 15.0:
        fatigue = max(fatigue, 0.8)

    temporal = _rolling_fatigue_state(average_ear, mar, pitch_tilt, settings, rolling_state=rolling_state)
    fatigue_index = temporal["fatigue_index"]
    fatigue = _clamp(fatigue_index / 100.0, 0.0, 1.0)

    return {
        "attention": round(_clamp(attention, 0.0, 100.0), 1),
        "fatigue": round(_clamp(fatigue, 0.0, 1.0), 3),
        "eye_engagement": round(_clamp(average_ear, 0.0, 0.5), 3),
        "posture_score": round(posture_score, 3),
        "microsleep": temporal["microsleep"],
        "yawning": temporal["yawning"],
        "drowsiness_state": temporal["drowsiness_state"],
        "fatigue_index": fatigue_index,
        "detection_thresholds": settings,
    }
