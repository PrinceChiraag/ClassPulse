import time
import json
import urllib.request
import urllib.error

import cv2
import mediapipe as mp

from classpulse.metrics import compute_attention_fatigue


class PrivacyPreservingAnalyzer:
    def __init__(self, backend_url="http://127.0.0.1:5000/api/telemetry"):
        self.backend_url = backend_url
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.frame_count = 0

    def _safe_landmarks(self, results):
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def _send_to_backend(self, metrics):
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **metrics,
        }
        request = urllib.request.Request(
            self.backend_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.read().decode("utf-8")
        except urllib.error.URLError:
            return None

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Unable to access webcam")

        print("ClassPulse CV pipeline started. Privacy-preserving landmarks only.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                if self.frame_count % 3 != 0:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_mesh.process(rgb_frame)
                pose_results = self.pose.process(rgb_frame)

                face_landmarks = self._safe_landmarks(face_results)
                pose_landmarks = pose_results.pose_landmarks
                metrics = compute_attention_fatigue(face_landmarks, pose_landmarks)

                self._send_to_backend(metrics)
                print(
                    f"Attention {metrics['attention']} | Fatigue {metrics['fatigue']} | "
                    f"EyeEngagement {metrics['eye_engagement']} | Posture {metrics['posture_score']}"
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
