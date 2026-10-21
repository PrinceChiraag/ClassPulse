# ClassPulse: Final-Year Project Brief

## Abstract

ClassPulse is a web-based prototype that explores whether lightweight computer-vision cues can help a teacher notice possible changes in classroom engagement during an on-demand scan. A responsive browser interface captures a frame with teacher permission and sends it to a Flask service. The service detects faces, estimates heuristic eye, posture, and fatigue signals, optionally detects phone-shaped objects with a pretrained object detector, and returns aggregate and per-face results. Teacher authentication, a versioned API, SQLite persistence, and container deployment demonstrate full-stack engineering around the vision pipeline. The prototype is intended for research demonstration and teacher reflection. Its outputs are uncertain visual cues, not ground truth about students.

## Problem and motivation

Instructors cannot continuously observe every learner while teaching. A simple, teacher-initiated snapshot may provide an additional room-level cue that invites a check-in or a change of activity. ClassPulse investigates the technical pipeline and interface for that interaction while minimizing default retention of camera data.

## Objectives

1. Build a responsive teacher portal with account management, scan controls, feedback, and settings.
2. Process browser-submitted images through face-landmark, posture, and optional object-detection components.
3. Report interpretable metrics and suggestions with explicit uncertainty and known limitations.
4. Provide authenticated web endpoints and a documented bearer-token API.
5. Persist teacher accounts and provide an explicit opt-in path for named scan records.
6. Package the service for repeatable deployment with persistent storage and a health endpoint.

## Architecture

```text
Teacher browser
  ├─ camera permission and on-demand capture
  ├─ responsive dashboard and scan settings
  └─ authenticated JSON requests
          │
          ▼
Flask application
  ├─ teacher session / mobile bearer authentication
  ├─ image validation and size limits
  ├─ MediaPipe landmarks → heuristic attention/fatigue/posture metrics
  ├─ OpenCV fallback when MediaPipe is unavailable
  ├─ Ultralytics phone-object detector (optional)
  └─ SQLite teacher accounts and explicitly saved student scans
          │
          ▼
Single-container deployment behind a TLS reverse proxy
```

Raw frame bytes are held in request memory for analysis. Named image records are saved only through the explicit opt-in workflow. SQLite and temporal face tracks are local to one app instance, so the deployment uses one worker and a persistent volume.

## Main technologies

- Python, Flask, SQLite, Gunicorn
- OpenCV, MediaPipe, NumPy, Ultralytics YOLO
- HTML, CSS, JavaScript, browser MediaDevices API, Progressive Web App manifest
- Docker and Docker Compose

## Evaluation plan

The repository does not contain a representative labeled classroom dataset, so do not report accuracy or classroom effectiveness as established facts. A defensible evaluation should be conducted with institutional approval and informed consent:

1. **Functional testing:** account creation/login/logout, access control, scan upload validation, camera permission flow, settings, mobile API authentication, and error states.
2. **Performance testing:** record end-to-end scan latency, memory use, and failure rates on declared hardware, image sizes, and lighting conditions.
3. **Vision component evaluation:** use a consented, representative, independently labeled dataset; report face/object precision and recall, and assess heuristic measures separately against a clearly defined reference protocol. Include subgroup and environmental error analysis where ethical and feasible.
4. **Usability evaluation:** ask instructors to complete defined tasks and collect task completion, error, and usability feedback. Do not collect student-level inferences as a proxy for learning outcomes.
5. **Privacy/security review:** verify access controls, secret handling, transport security, database backups/retention, and that default scans do not create student image records.

## Limitations and responsible use

- Eye direction, posture, face presence, and yawning do not establish attention, comprehension, fatigue, or wellbeing.
- The fallback score is a crude image heuristic, not a trained or validated behavioral model.
- The general-purpose phone detector may miss small or occluded objects and can misclassify similar objects; it cannot establish phone ownership or use.
- Lighting, camera placement, occlusion, facial variation, and demographic representation can change system performance.
- Face detections should not be used to identify or rank students. Per-face labels are temporary positions in a frame, not identities.
- Student data handling requires local institutional review, consent, a retention schedule, and restricted access. The prototype does not implement automated record expiry or a school records integration.
- The project is a single-instance prototype. High availability, distributed sessions, PostgreSQL, audit logs, and operational alerting are future engineering work.

## Suggested report and demonstration

1. Introduction, problem definition, and scoped claims
2. Related work and responsible-use considerations
3. Requirements and architecture
4. Computer-vision methods and metric definitions
5. Web/API implementation and privacy controls
6. Evaluation protocol, results, and failure analysis
7. Limitations, conclusion, and future work

For the demo, show a local teacher login, explain browser camera permission, capture a consented test image, review the result and uncertainty, show that saving is opt-in, then demonstrate the health endpoint and container configuration. Use synthetic or specifically consented imagery.
