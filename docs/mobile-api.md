# ClassPulse mobile API

The web dashboard uses the same versioned HTTP API intended for a future Android or iOS client. A native client should capture frames locally with the platform camera API and send JPEG frames to the scan endpoint; the Flask server performs inference and returns JSON. No browser `getUserMedia` code is required in the mobile client.

## Authentication

1. `POST /api/v1/auth/login` with `{"email":"...","password":"..."}`.
2. Keep the returned `access_token` in the platform's secure credential store.
3. Send `Authorization: Bearer <access_token>` on every other `/api/v1` request.
4. Tokens expire after 12 hours. The logout endpoint is stateless; delete the token from secure storage. Sign in again for a new token.

Set a stable, private `SECRET_KEY` in the server environment before deploying. Tokens and browser sessions are signed with this key. Serve the API over HTTPS outside localhost.

## Scan a frame

`POST /api/v1/scan` with JSON. `image_data` may be a base64 JPEG string or a data URL. Requests are limited by the 8 MiB application request cap; decoded images above 12 megapixels are rejected. Scan and dashboard endpoints require teacher authentication.

```json
{
  "image_data": "data:image/jpeg;base64,...",
  "detection_mode": "multi-face",
  "detection_settings": {
    "fatigue_detection": true,
    "posture_analysis": true
  }
}
```

The response includes heuristic face metrics, temporary per-face boxes and fatigue cues, plus `phone_detections`. Each phone detection has a label, confidence, and pixel-space `bbox` (`x`, `y`, `width`, `height`) measured against the submitted image. `phone_detection_status` is `ready`, `unavailable`, `no-phone-class`, or `error`. `phone_alert` indicates that the model found one or more phone objects; it does not identify a person or establish phone use. Treat all outputs as uncertain prompts for teacher review, not measures of learning or grounds for decisions about an individual student.

Other bearer-protected endpoints:

- `GET /api/v1/auth/me`
- `POST /api/v1/auth/logout`
- `GET /api/v1/dashboard-config`
- `GET /api/v1/telemetry`
- `GET /api/v1/students`
- `GET /api/v1/alerts`

The API accepts normal native HTTPS clients. CORS is not enabled because native Android and iOS clients do not use browser CORS; a future browser-hosted client should use a narrowly configured origin allowlist.

## Mobile client responsibilities

- Request camera permission only after the user starts a scan; explain why the camera is needed.
- Send resized JPEG frames at a modest interval rather than uploading a full video stream.
- Keep the bearer token in Keychain (iOS) or an encrypted credential store (Android), never in source code or ordinary preferences.
- Draw returned phone and face boxes over the local preview by scaling the pixel-space boxes to the preview dimensions.
- Clear local frames after sending and show a visible stop-camera control.
