const STORAGE_KEY = "classpulse-config";

if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => navigator.serviceWorker.register("/service-worker.js").catch(() => {}));
}

const summarySelectors = {
    average_attention: document.getElementById("avg-attention"),
    fatigue_alerts: document.getElementById("fatigue-alerts"),
    active_students: document.getElementById("active-students"),
    room_status: document.getElementById("room-status"),
    status_detail: document.getElementById("status-detail"),
    attention_trend: document.getElementById("attention-trend"),
    fatigue_trend: document.getElementById("fatigue-trend"),
    eye_engagement: document.getElementById("eye-engagement"),
    posture_score: document.getElementById("posture-score"),
    last_refresh: document.getElementById("last-refresh"),
    hero_health: document.getElementById("hero-health"),
    hero_focus: document.getElementById("hero-focus"),
    hero_snapshot: document.getElementById("hero-snapshot"),
    face_status: document.getElementById("face-detail-status"),
    phone_warning: document.getElementById("face-phone-warning"),
    face_risk: document.getElementById("face-detail-risk"),
    face_attention: document.getElementById("face-detail-attention"),
    face_eye: document.getElementById("face-detail-eye"),
    face_posture: document.getElementById("face-detail-posture"),
    face_fatigue: document.getElementById("face-detail-fatigue"),
    face_confidence: document.getElementById("face-detail-confidence"),
    confidence_bar: document.getElementById("confidence-bar"),
    overall_score_badge: document.getElementById("overall-score-badge"),
    drowsiness_status_badge: document.getElementById("drowsiness-status-badge"),
    score_badges: document.getElementById("score-badges"),
    face_recommendation: document.getElementById("face-detail-recommendation"),
    focus_stability: document.getElementById("focus-stability"),
    face_coverage: document.getElementById("face-coverage"),
    last_capture: document.getElementById("last-capture"),
    camera_readiness: document.getElementById("camera-readiness"),
    student_count: document.getElementById("student-count"),
    recommended_action: document.getElementById("recommended-action"),
};

const historyBody = document.getElementById("history-body");
const cameraToggle = document.getElementById("camera-toggle");
const capturePhotoButton = document.getElementById("capture-photo");
const clearPhotoButton = document.getElementById("clear-photo");
const syncDashboardButton = document.getElementById("sync-dashboard");
const logoutButton = document.getElementById("logout-button");
const cameraFeed = document.getElementById("camera-feed");
const phoneOverlay = document.getElementById("phone-overlay");
const phoneDetectionStatus = document.getElementById("phone-detection-status");
const phoneScanAlert = document.getElementById("phone-scan-alert");
const photoPanel = document.getElementById("photo-panel");
const capturedPhoto = document.getElementById("captured-photo");
const cameraStatus = document.getElementById("camera-status");
const analysisModeSelect = document.getElementById("analysis-mode");
const cameraSourceSelect = document.getElementById("camera-source");
const cameraFacingButtons = document.querySelectorAll("[data-facing-camera]");
const detectionModeSelect = document.getElementById("detection-mode");
const earBlinkCutoffInput = document.getElementById("ear-blink-cutoff");
const marYawnSensitivityInput = document.getElementById("mar-yawn-sensitivity");
const postureTiltDegreesInput = document.getElementById("posture-tilt-degrees");
const alertThresholdInput = document.getElementById("alert-threshold");
const alertThresholdLabel = document.getElementById("alert-threshold-label");
const phoneConfidenceInput = document.getElementById("phone-confidence");
const phoneConfidenceLabel = document.getElementById("phone-confidence-label");
const earBlinkCutoffLabel = document.getElementById("ear-blink-cutoff-label");
const marYawnSensitivityLabel = document.getElementById("mar-yawn-sensitivity-label");
const postureTiltDegreesLabel = document.getElementById("posture-tilt-degrees-label");
const cameraContext = document.getElementById("camera-context");
const studentRoster = document.getElementById("student-roster");
const heatmap = document.getElementById("heatmap");
const alertList = document.getElementById("alert-list");
const optionControls = document.getElementById("option-controls");
const cameraFrame = document.querySelector(".camera-frame");
const scanProgressOverlay = document.getElementById("scan-progress-overlay");
const cameraAccessBanner = document.getElementById("camera-access-banner");

let localStream = null;
let privacyWireframeEnabled = false;
let dashboardRefreshTimer = null;
const scoreHistory = [];
let dashboardConfig = {
    controls: {
        camera_feed: true,
        student_tracking: true,
        fatigue_detection: true,
        posture_analysis: true,
        privacy_mode: true,
        face_blur: false,
        preview_blur: false,
        phone_detection: true,
        phone_confidence: 0.12,
        show_phone_boxes: true,
        sound_alerts: false,
        auto_refresh: true,
        detection_mode: "multi-face",
        ear_blink_cutoff: 0.18,
        mar_yawn_sensitivity: 0.24,
        posture_tilt_degrees: 18,
        alert_threshold: 0.7,
        camera_source: "default",
    },
    analysis_modes: ["Lecture mode", "Exam mode", "Group review", "Hands-on lab"],
    default_mode: "Lecture mode",
    camera_devices: [{ id: "default", label: "Default laptop camera" }],
    student_roster: [],
};

function saveConfigToStorage() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(dashboardConfig));
    } catch (error) {
        console.warn("Unable to persist settings", error);
    }
}

function loadConfigFromStorage() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            return;
        }
        const saved = JSON.parse(raw);
        if (saved && saved.controls) {
            dashboardConfig = { ...dashboardConfig, ...saved, controls: { ...dashboardConfig.controls, ...saved.controls } };
            // Older releases used face_blur/privacy_mode to blur the entire preview.
            // Keep the live scanner clear unless the new preview-only control is enabled.
            if (saved.controls.preview_blur === undefined) {
                dashboardConfig.controls.preview_blur = false;
            }
        }
    } catch (error) {
        console.warn("Unable to load stored settings", error);
    }
}

function updateStatusSummary() {
    const tracking = document.getElementById("tracking-status");
    const fatigue = document.getElementById("fatigue-status");
    const privacy = document.getElementById("privacy-status");
    const refresh = document.getElementById("refresh-status");

    if (tracking) tracking.textContent = dashboardConfig.controls.student_tracking ? "On" : "Off";
    if (fatigue) fatigue.textContent = dashboardConfig.controls.fatigue_detection ? "On" : "Off";
    if (privacy) privacy.textContent = dashboardConfig.controls.privacy_mode ? "On" : "Off";
    if (refresh) refresh.textContent = dashboardConfig.controls.auto_refresh ? "On" : "Off";
}

function formatMetric(value) {
    return value === null || value === undefined ? "--" : value;
}

async function loadCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        return;
    }

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((device) => device.kind === "videoinput");
        if (!videoInputs.length) {
            return;
        }

        dashboardConfig.camera_devices = [
            { id: "default", label: "Default laptop camera" },
            ...videoInputs.map((device) => ({
                id: device.deviceId,
                label: device.label || `Camera ${device.deviceId.slice(0, 6)}`,
            })),
        ];

        renderControls();
    } catch (error) {
        console.warn("Unable to enumerate camera devices", error);
    }
}

function renderControls() {
    const controls = Object.entries(dashboardConfig.controls || {});
    const checkboxes = [...document.querySelectorAll("[data-option]")];

    if (document.getElementById("focus-mode")) {
        const selectedMode = dashboardConfig.controls.focus_mode || "lecture";
        document.getElementById("focus-mode").value = selectedMode;
    }

    checkboxes.forEach((checkbox) => {
        const key = checkbox.dataset.option;
        const isEnabled = dashboardConfig.controls[key] !== undefined ? dashboardConfig.controls[key] : true;
        checkbox.checked = Boolean(isEnabled);
        checkbox.disabled = false;
    });

    if (analysisModeSelect) {
        analysisModeSelect.innerHTML = "";
        dashboardConfig.analysis_modes.forEach((mode) => {
            const option = document.createElement("option");
            option.value = mode;
            option.textContent = mode;
        option.selected = mode === (dashboardConfig.controls.analysis_mode || dashboardConfig.default_mode);
            analysisModeSelect.appendChild(option);
        });
    }

    if (cameraSourceSelect) {
        cameraSourceSelect.innerHTML = "";
        dashboardConfig.camera_devices.forEach((device) => {
            const option = document.createElement("option");
            option.value = device.id;
            option.textContent = device.label;
            option.selected = device.id === (dashboardConfig.controls.camera_source || "default");
            cameraSourceSelect.appendChild(option);
        });
    }
    updateCameraFacingButtons();

    if (detectionModeSelect) {
        const modeValue = dashboardConfig.controls.detection_mode || "multi-face";
        detectionModeSelect.value = modeValue;
    }

    if (phoneConfidenceInput) {
        const value = Number(dashboardConfig.controls.phone_confidence ?? 0.12);
        phoneConfidenceInput.value = String(value);
        if (phoneConfidenceLabel) phoneConfidenceLabel.textContent = value.toFixed(2);
    }

    if (earBlinkCutoffInput) {
        const value = Number(dashboardConfig.controls.ear_blink_cutoff ?? 0.18);
        earBlinkCutoffInput.value = String(value);
        if (earBlinkCutoffLabel) {
            earBlinkCutoffLabel.textContent = value.toFixed(2);
        }
    }

    if (marYawnSensitivityInput) {
        const value = Number(dashboardConfig.controls.mar_yawn_sensitivity ?? 0.24);
        marYawnSensitivityInput.value = String(value);
        if (marYawnSensitivityLabel) {
            marYawnSensitivityLabel.textContent = value.toFixed(2);
        }
    }

    if (postureTiltDegreesInput) {
        const value = Number(dashboardConfig.controls.posture_tilt_degrees ?? 18);
        postureTiltDegreesInput.value = String(value);
        if (postureTiltDegreesLabel) {
            postureTiltDegreesLabel.textContent = `${value.toFixed(0)}°`;
        }
    }

    if (alertThresholdInput) {
        const thresholdValue = Number(dashboardConfig.controls.alert_threshold ?? 0.7);
        alertThresholdInput.value = String(thresholdValue);
        if (alertThresholdLabel) {
            alertThresholdLabel.textContent = thresholdValue.toFixed(2);
        }
    }

    if (cameraContext) {
        cameraContext.textContent = `${analysisModeSelect?.value || dashboardConfig.default_mode} feed`;
    }

    applyCameraOptions();

    if (controls.length === 0 && optionControls) {
        optionControls.style.display = "none";
    }

    updateStatusSummary();
}

function applyCameraOptions() {
    if (cameraFeed) {
        cameraFeed.style.display = dashboardConfig.controls.camera_feed === false ? "none" : "block";
        cameraFeed.style.filter = dashboardConfig.controls.preview_blur ? "blur(3px)" : "none";
    }
    if (studentRoster && dashboardConfig.controls.student_tracking === false) {
        studentRoster.innerHTML = '<div class="empty-state">Student tracking is paused. Turn it on in settings to show per-face results.</div>';
    }
}

function configureLiveTimers() {
    if (dashboardRefreshTimer) clearInterval(dashboardRefreshTimer);
    dashboardRefreshTimer = dashboardConfig.controls.auto_refresh === false ? null : setInterval(refreshDashboard, 5000);
}

function renderPhoneDetections(result) {
    const detections = Array.isArray(result.phone_detections) ? result.phone_detections : [];
    const state = result.phone_detection_status || "unavailable";
    if (summarySelectors.phone_warning) {
        summarySelectors.phone_warning.hidden = detections.length === 0;
    }
    if (phoneScanAlert) {
        phoneScanAlert.hidden = detections.length === 0;
        phoneScanAlert.textContent = detections.length
            ? `PHONE DETECTED: ${detections.length} phone${detections.length === 1 ? "" : "s"} found in frame.`
            : "";
    }
    if (phoneOverlay && cameraFeed) {
        const frameWidth = Number(result.frame_width || cameraFeed.videoWidth || 640);
        const frameHeight = Number(result.frame_height || cameraFeed.videoHeight || 480);
        const rect = cameraFeed.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.round(rect.width * dpr));
        const height = Math.max(1, Math.round(rect.height * dpr));
        if (phoneOverlay.width !== width || phoneOverlay.height !== height) {
            phoneOverlay.width = width;
            phoneOverlay.height = height;
        }
        const ctx = phoneOverlay.getContext("2d");
        ctx.clearRect(0, 0, width, height);
        const scale = Math.max(width / frameWidth, height / frameHeight);
        const offsetX = (width - frameWidth * scale) / 2;
        const offsetY = (height - frameHeight * scale) / 2;
        if (dashboardConfig.controls.show_phone_boxes !== false) detections.forEach((item) => {
            const box = item.bbox || {};
            const y = offsetY + Number(box.y || 0) * scale;
            const w = Number(box.width || 0) * scale, h = Number(box.height || 0) * scale;
            const rawX = Number(box.x || 0);
            const sourceX = cameraFeed.classList.contains("camera-mirrored") ? frameWidth - rawX - Number(box.width || 0) : rawX;
            const x = offsetX + sourceX * scale;
            ctx.lineWidth = Math.max(2 * dpr, width / 500);
            ctx.strokeStyle = "#ff3030";
            ctx.fillStyle = "rgba(255, 48, 48, .20)";
            ctx.strokeRect(x, y, w, h);
            ctx.fillRect(x, y, w, h);
            ctx.font = `${Math.max(14 * dpr, width / 55)}px Segoe UI, sans-serif`;
            const label = `PHONE DETECTED ${Math.round(Number(item.confidence || 0) * 100)}%`;
            const labelWidth = ctx.measureText(label).width + 12;
            ctx.fillStyle = "#b42318";
            ctx.fillRect(x, Math.max(0, y - 28 * dpr), labelWidth, 26 * dpr);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, x + 6 * dpr, Math.max(19 * dpr, y - 9 * dpr));
        });
    }
    if (phoneDetectionStatus) {
        phoneDetectionStatus.classList.remove("detected", "unavailable");
        if (detections.length) {
            phoneDetectionStatus.classList.add("detected");
            phoneDetectionStatus.textContent = `⚠ ${detections.length} phone object${detections.length === 1 ? "" : "s"} detected · check context`;
        } else if (state === "ready") {
            phoneDetectionStatus.textContent = "Phone scan active · no phone objects detected";
        } else if (state === "fallback") {
            phoneDetectionStatus.classList.add("detected");
            phoneDetectionStatus.textContent = "Fallback phone scan active";
        } else {
            phoneDetectionStatus.classList.add("unavailable");
            phoneDetectionStatus.textContent = state === "disabled"
                ? "Phone scan is turned off"
                : state === "no-phone-class"
                    ? "Phone detection unavailable: model has no phone class"
                    : state === "not-installed"
                        ? "Phone detector not installed in the server Python environment; install requirements.txt and restart"
                        : state === "model-not-found"
                            ? "Phone detector weights are missing; set CLASSPULSE_PHONE_MODEL to yolo11n.pt"
                    : state === "error"
                        ? "Phone detector failed on the server; check deployment logs"
                        : state === "load-error"
                            ? "Phone detector could not load; check server logs and model compatibility"
                            : "Phone detection unavailable on the server; verify detector and model weights";
        }
    }
    if (alertList) {
        alertList.querySelector("[data-phone-alert]")?.remove();
        if (detections.length) {
            const item = document.createElement("li");
            item.className = "alert-item warning";
            item.dataset.phoneAlert = "true";
            item.innerHTML = `<strong>Possible phone use</strong><span>${detections.length} phone object${detections.length === 1 ? "" : "s"} in frame</span><small>Visual cue only; check context</small>`;
            alertList.prepend(item);
        }
    }
}

function updatePortalDetails(config = {}) {
    const details = config.portal_details || {};
    if (summarySelectors.focus_stability) {
        summarySelectors.focus_stability.textContent = details.focus_stability || "Stable";
    }
    if (summarySelectors.face_coverage) {
        summarySelectors.face_coverage.textContent = details.face_coverage || "Waiting for live scan";
    }
    if (summarySelectors.last_capture) {
        summarySelectors.last_capture.textContent = details.last_capture || "No scan yet";
    }
    if (summarySelectors.camera_readiness) {
        summarySelectors.camera_readiness.textContent = details.camera_status || "Ready for live scan";
    }
    if (summarySelectors.student_count) {
        summarySelectors.student_count.textContent = `${Number(details.student_count || 0)} detected`;
    }
    if (summarySelectors.recommended_action) {
        summarySelectors.recommended_action.textContent = details.recommended_action || "Keep the current teaching rhythm and monitor attention drift in the back row.";
    }
}

function updateDrowsinessBadge(drowsinessState = "Alert") {
    if (!summarySelectors.drowsiness_status_badge) return;

    const normalized = String(drowsinessState || "Alert").toLowerCase();
    summarySelectors.drowsiness_status_badge.textContent = drowsinessState;
    if (normalized.includes("yawn")) {
        summarySelectors.drowsiness_status_badge.className = "status-badge warning";
    } else if (normalized.includes("drows")) {
        summarySelectors.drowsiness_status_badge.className = "status-badge danger";
    } else {
        summarySelectors.drowsiness_status_badge.className = "status-badge alert";
    }
}

function renderSparkline(values) {
    const sparkline = document.getElementById("score-sparkline");
    if (!sparkline) return;

    const points = values.length ? values : [50, 52, 48, 55, 57, 53, 60, 58, 63, 61];
    const width = 120;
    const height = 40;
    const min = Math.min(...points, 0);
    const max = Math.max(...points, 100);

    const path = points
        .map((value, index) => {
            const x = (index / Math.max(points.length - 1, 1)) * width;
            const y = height - ((value - min) / Math.max(max - min, 1)) * (height - 6) - 3;
            return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ");

    sparkline.innerHTML = `<path d="${path}" fill="none" stroke="rgba(88, 212, 255, 0.95)" stroke-width="2.2" stroke-linecap="round" />`;
}

function togglePrivacyWireframe() {
    privacyWireframeEnabled = !privacyWireframeEnabled;
    cameraFrame?.classList.toggle("privacy-view", privacyWireframeEnabled);
    const button = document.getElementById("privacy-wireframe-toggle");
    if (button) button.textContent = privacyWireframeEnabled ? "Hide scan guide" : "Show scan guide";
}

function renderStudentRoster(students) {
    if (!studentRoster) return;
    if (dashboardConfig.controls.student_tracking === false) {
        studentRoster.innerHTML = '<div class="empty-state">Student tracking is paused. Turn it on in settings to show per-face results.</div>';
        return;
    }

    const roster = students && students.length ? students : dashboardConfig.student_roster || [];
    studentRoster.innerHTML = roster.length
        ? roster
              .map((student) => {
                  const name = student.name || student.label || "Student";
                  const status = student.status || (Number(student.attention ?? 0) >= 80 ? "Focused" : "Review");
                  const attention = Number(student.attention ?? 0);
                  const fatigue = Number(student.fatigue ?? 0);
                  return `
                      <div class="student-card ${status === "Focused" || status === "Alert" ? "focused" : "alert"}">
                          <div>
                              <strong>${name}</strong>
                              <span>${status}</span>
                          </div>
                          <div class="student-score">
                              <span>${attention.toFixed(1)}%</span>
                              <small>Fatigue: ${(fatigue * 100).toFixed(1)}%</small>
                          </div>
                      </div>
                  `;
              })
              .join("")
        : '<div class="empty-state">No faces detected yet. Capture a frame to review temporary face-level cues. Named records are saved only when you explicitly opt in.</div>';
}

function renderHeatmap(zones = []) {
    if (!heatmap) return;

    const fallbackZones = [
        { name: "Front row", attention: 0, severity: "low" },
        { name: "Middle row", attention: 0, severity: "medium" },
        { name: "Back row", attention: 0, severity: "high" },
    ];
    const items = zones.length ? zones : fallbackZones;

    heatmap.innerHTML = items
        .map(
            (zone) => `
                <div class="heatmap-zone ${zone.severity || "medium"}">
                    <span>${zone.name}</span>
                    <strong>${Number(zone.attention ?? 0).toFixed(0)}%</strong>
                </div>
            `,
        )
        .join("");
}

async function loadAlerts() {
    try {
        const response = await fetch("/api/alerts");
        if (!response.ok) {
            throw new Error("Failed to fetch alerts");
        }

        const payload = await response.json();
        const alerts = payload.alerts || [];
        if (alertList) {
            alertList.innerHTML = alerts
                .map(
                    (alert) => `
                        <li class="alert-item ${alert.severity === "low" ? "good" : "warning"}">
                            <strong>${alert.student}</strong>
                            <span>${alert.reason}</span>
                            <small>${alert.attention}% attention</small>
                        </li>
                    `,
                )
                .join("");
        }

        if (payload.heatmap && payload.heatmap.zones && heatmap) {
            heatmap.innerHTML = payload.heatmap.zones
                .map(
                    (zone) => `
                        <div class="heatmap-zone ${zone.severity}">
                            <span>${zone.name}</span>
                            <strong>${zone.attention}%</strong>
                        </div>
                    `,
                )
                .join("");
        }
    } catch (error) {
        console.error(error);
    }
}

function renderAlertList(summary, history) {
    if (!alertList) return;

    const threshold = Number(dashboardConfig.controls.alert_threshold ?? 0.7);
    const fatigueAlerts = history.filter((entry) => Number(entry.fatigue) > threshold);
    const alerts = fatigueAlerts.length
        ? fatigueAlerts.slice(-3).map((entry, index) => {
              const label = dashboardConfig.student_roster[index]?.name || `Student ${index + 1}`;
              return `${label} fatigue risk: ${(Number(entry.fatigue) * 100).toFixed(0)}%`;
          })
        : [
              `No fatigue alerts detected above ${threshold.toFixed(2)} in ${summary.active_students || 0} tracked sessions.`,
          ];

    alertList.innerHTML = alerts
        .map(
            (alert) => `
                <li class="alert-item ${alert.includes("No fatigue") ? "good" : "warning"}">${alert}</li>
            `,
        )
        .join("");
}

function updateStatusFromMetrics(summary) {
    if (!summarySelectors.room_status || !summarySelectors.status_detail || !summarySelectors.attention_trend || !summarySelectors.fatigue_trend) {
        return;
    }

    const attention = Number(summary.average_attention || 0);
    const fatigue = Number(summary.fatigue_alerts || 0);

    if (attention >= 80) {
        summarySelectors.room_status.textContent = "Strong";
        summarySelectors.status_detail.textContent = "High engagement";
        summarySelectors.attention_trend.textContent = "Excellent";
        summarySelectors.attention_trend.className = "trend positive";
        if (summarySelectors.hero_health) summarySelectors.hero_health.textContent = "Excellent";
        if (summarySelectors.hero_focus) summarySelectors.hero_focus.textContent = "Excellent";
    } else if (attention >= 60) {
        summarySelectors.room_status.textContent = "Stable";
        summarySelectors.status_detail.textContent = "Moderate focus";
        summarySelectors.attention_trend.textContent = "Steady";
        summarySelectors.attention_trend.className = "trend neutral";
        if (summarySelectors.hero_health) summarySelectors.hero_health.textContent = "Stable";
        if (summarySelectors.hero_focus) summarySelectors.hero_focus.textContent = "Steady";
    } else {
        summarySelectors.room_status.textContent = "Watch";
        summarySelectors.status_detail.textContent = "Needs attention";
        summarySelectors.attention_trend.textContent = "Low focus";
        summarySelectors.attention_trend.className = "trend neutral";
        if (summarySelectors.hero_health) summarySelectors.hero_health.textContent = "Review";
        if (summarySelectors.hero_focus) summarySelectors.hero_focus.textContent = "Low";
    }

    if (fatigue > 0) {
        summarySelectors.fatigue_trend.textContent = "Alerts active";
        summarySelectors.fatigue_trend.className = "trend neutral";
        if (summarySelectors.hero_snapshot) summarySelectors.hero_snapshot.textContent = "Alert";
    } else {
        summarySelectors.fatigue_trend.textContent = "Healthy";
        summarySelectors.fatigue_trend.className = "trend positive";
        if (summarySelectors.hero_snapshot) summarySelectors.hero_snapshot.textContent = "Clear";
    }
}

async function loadDashboardConfig() {
    try {
        const response = await fetch("/api/dashboard-config");
        if (response.ok) {
            const config = await response.json();
            const savedControls = { ...dashboardConfig.controls };
            const localControls = { ...savedControls };
            for (const [key, value] of Object.entries(config.controls || {})) {
                if (localStorage.getItem(STORAGE_KEY) === null || localControls[key] === undefined) localControls[key] = value;
            }
            dashboardConfig = { ...dashboardConfig, ...config, controls: { ...config.controls, ...localControls }, default_mode: localControls.analysis_mode || config.default_mode };
            renderControls();
            updatePortalDetails(config);
            renderStudentRoster(config.student_roster || dashboardConfig.student_roster);
        }
    } catch (error) {
        console.error("Unable to load dashboard config", error);
    }
}

async function loadStudentSnapshot() {
    try {
        const response = await fetch("/api/students");
        if (!response.ok) {
            throw new Error("Failed to fetch student snapshot");
        }

        const payload = await response.json();
        const students = payload.students || [];
        if (students.length) {
            dashboardConfig.student_roster = students;
            renderStudentRoster(students);
        }
    } catch (error) {
        console.error(error);
    }
}

async function refreshDashboard() {
    try {
        const response = await fetch("/api/telemetry");
        if (!response.ok) {
            throw new Error("Failed to fetch telemetry");
        }

        const payload = await response.json();
        const summary = payload.summary || {};
        const history = payload.history || [];

        if (summarySelectors.average_attention) {
            summarySelectors.average_attention.textContent = `${formatMetric(summary.average_attention)}%`;
        }
        if (summarySelectors.fatigue_alerts) {
            summarySelectors.fatigue_alerts.textContent = formatMetric(summary.fatigue_alerts);
        }
        if (summarySelectors.active_students) {
            summarySelectors.active_students.textContent = formatMetric(history.length);
        }
        if (summarySelectors.eye_engagement) {
            summarySelectors.eye_engagement.textContent = `${formatMetric(Number(history.at(-1)?.eye_engagement || 0) * 100)}%`;
        }
        if (summarySelectors.posture_score) {
            summarySelectors.posture_score.textContent = `${formatMetric(Number(history.at(-1)?.posture_score || 0) * 100)}%`;
        }
        if (summarySelectors.last_refresh) {
            summarySelectors.last_refresh.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }

        updateStatusFromMetrics(summary);
        renderAlertList(summary, history);
        await loadStudentSnapshot();
        await loadAlerts();

        if (historyBody) {
            historyBody.innerHTML = "";
            history.slice(-10).reverse().forEach((entry) => {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${entry.timestamp}</td>
                    <td>${entry.attention}</td>
                    <td>${entry.fatigue}</td>
                `;
                historyBody.appendChild(row);
            });
        }
    } catch (error) {
        console.error(error);
    }
}

function setCameraUnavailable(message = "Camera unavailable. Connect a webcam and allow browser access.") {
    if (cameraStatus) {
        cameraStatus.textContent = "Camera unavailable";
        cameraStatus.className = "status-pill idle";
    }
    if (cameraToggle) {
        cameraToggle.textContent = "Start camera";
    }
    if (cameraContext) {
        cameraContext.textContent = message;
    }
    if (cameraAccessBanner) {
        const bannerText = message.includes("secure context")
            ? "Use localhost or HTTPS in the browser, then allow camera access when prompted."
            : "Allow camera access in the browser and confirm the webcam permission request.";
        cameraAccessBanner.innerHTML = `
            <div class="camera-access-icon">📷</div>
            <div>
                <strong>Allow camera access</strong>
                <p>${bannerText}</p>
            </div>
        `;
    }
    if (cameraFrame) {
        cameraFrame.classList.add("camera-error");
    }
    if (cameraFeed) {
        cameraFeed.style.display = dashboardConfig.controls.camera_feed === false ? "none" : "block";
    }
    if (summarySelectors.face_status) {
        summarySelectors.face_status.textContent = "Camera blocked";
    }
    if (summarySelectors.face_recommendation) {
        summarySelectors.face_recommendation.textContent = "Allow camera access in the browser to run a full face assessment.";
    }
}

function getScoreBadgeClass(value) {
    if (value >= 75) return "good";
    if (value >= 50) return "warning";
    return "danger";
}

function renderFaceDetails(payload) {
    const detail = payload || {};
    const faceEntries = Array.isArray(detail.faces) ? detail.faces : (detail.face_detected ? [detail] : []);
    const primaryFace = faceEntries[0] || {};
    const hasFace = faceEntries.length > 0;
    const statusText = ["basic-fallback", "face-detection-fallback"].includes(detail.analysis_mode) && hasFace
        ? "Basic estimate · detailed face analysis unavailable"
        : detail.status || (hasFace ? (faceEntries.length > 1 ? `Multi-face scan: ${faceEntries.length} students` : "Face analyzed") : "No clear face detected");
    const confidenceValue = Number(primaryFace.confidence ?? detail.confidence ?? 0);
    const confidencePercent = Math.max(0, Math.min(100, confidenceValue));
    const eyeScore = Number(primaryFace.eye_engagement ?? detail.eye_engagement ?? 0) * 100;
    const postureScore = Number(primaryFace.posture_score ?? detail.posture_score ?? 0) * 100;
    const fatigueScore = Number(primaryFace.fatigue_index ?? Number(primaryFace.fatigue ?? detail.fatigue ?? 0) * 100);
    const attentionScore = Number(primaryFace.attention ?? detail.attention ?? 0);
    const overallScore = Number(primaryFace.overall_score ?? detail.overall_score ?? 0);
    const drowsinessState = primaryFace.drowsiness_state || detail.drowsiness_state || "Alert";

    if (!hasFace) {
        if (studentRoster) studentRoster.innerHTML = '<div class="empty-state">No face detected. Improve lighting, face the camera, and capture again.</div>';
        if (heatmap) heatmap.innerHTML = '<div class="empty-state">A face scan is needed to show attention.</div>';
        if (alertList) alertList.innerHTML = '<li class="alert-item warning"><strong>No face detected</strong><span>Scan not analyzed</span><small>Adjust the frame and try again.</small></li>';
    }

    if (!hasFace) {
        // The empty state is rendered above; clear any stale roster from an earlier scan.
        renderStudentRoster([]);
        renderHeatmap([]);
        if (alertList) alertList.innerHTML = '<li class="alert-item warning"><strong>No face detected</strong><span>Scan not analyzed</span><small>Improve lighting and keep your face centered.</small></li>';
    } else if (faceEntries.length > 1) {
        const roster = faceEntries.map((face, index) => ({
            name: face.label || `Student ${index + 1}`,
            status: face.drowsiness_state || (Number(face.attention ?? 0) >= 80 ? "Focused" : "Review"),
            attention: Number(face.attention ?? 0),
            fatigue: Number(face.fatigue ?? 0),
        }));
        renderStudentRoster(roster);
        renderHeatmap(
            roster.map((student, index) => ({
                name: student.name,
                attention: student.attention,
                severity: student.attention >= 80 ? "low" : student.attention >= 60 ? "medium" : "high",
            })),
        );
        if (alertList) {
            alertList.innerHTML = roster
                .map(
                    (student) => `
                        <li class="alert-item ${Number(student.attention) >= 70 ? "good" : "warning"}">
                            <strong>${student.name}</strong>
                            <span>${student.status}</span>
                            <small>${Number(student.attention).toFixed(1)}% attention</small>
                        </li>
                    `,
                )
                .join("");
        }
    } else {
        const singleRoster = faceEntries.map((face, index) => ({
            name: face.label || `Student ${index + 1}`,
            status: face.drowsiness_state || (Number(face.attention ?? 0) >= 80 ? "Focused" : "Review"),
            attention: Number(face.attention ?? 0),
            fatigue: Number(face.fatigue ?? 0),
        }));
        renderStudentRoster(singleRoster);
        renderHeatmap(singleRoster.map((student) => ({
            name: student.name,
            attention: student.attention,
            severity: student.status === "Drowsy" || student.status === "Yawning" ? "high" : student.attention >= 80 ? "low" : "medium",
        })));
        if (alertList) {
            alertList.innerHTML = singleRoster.map((student) => `
                <li class="alert-item ${student.status === "Drowsy" || student.status === "Yawning" ? "warning" : "good"}">
                    <strong>${student.name}</strong><span>${student.status}</span>
                    <small>${student.attention.toFixed(1)}% attention</small>
                </li>`).join("");
        }
    }

    if (summarySelectors.face_status) {
        summarySelectors.face_status.textContent = statusText;
    }
    if (summarySelectors.face_risk) {
        summarySelectors.face_risk.textContent = detail.risk_level || (hasFace ? "Moderate" : "Needs reset");
    }
    if (summarySelectors.face_attention) {
        summarySelectors.face_attention.textContent = hasFace ? `${attentionScore.toFixed(1)}/100` : "Awaiting face";
    }
    if (summarySelectors.face_eye) {
        summarySelectors.face_eye.textContent = hasFace ? `${eyeScore.toFixed(1)}/100` : "Awaiting face";
    }
    if (summarySelectors.face_posture) {
        summarySelectors.face_posture.textContent = hasFace ? `${postureScore.toFixed(1)}/100` : "Awaiting face";
    }
    if (summarySelectors.face_fatigue) {
        summarySelectors.face_fatigue.textContent = hasFace ? `${fatigueScore.toFixed(1)}%` : "Awaiting face";
    }
    if (summarySelectors.face_confidence) {
        summarySelectors.face_confidence.textContent = hasFace ? `${overallScore.toFixed(1)}/100` : "Awaiting face";
    }
    if (summarySelectors.overall_score_badge) {
        summarySelectors.overall_score_badge.textContent = hasFace ? `${overallScore.toFixed(1)}/100` : "Awaiting face";
    }
    updateDrowsinessBadge(drowsinessState);
    if (scoreHistory.length > 60) scoreHistory.shift();
    if (hasFace) scoreHistory.push(overallScore);
    renderSparkline(scoreHistory);
    if (summarySelectors.score_badges) {
        const badges = hasFace ? [
            { label: `Attention`, value: attentionScore, css: getScoreBadgeClass(attentionScore) },
            { label: `Eyes`, value: eyeScore, css: getScoreBadgeClass(eyeScore) },
            { label: `Posture`, value: postureScore, css: getScoreBadgeClass(postureScore) },
            { label: `Fatigue`, value: fatigueScore, css: getScoreBadgeClass(100 - fatigueScore) },
        ] : [];
        summarySelectors.score_badges.innerHTML = badges
            .map(
                (badge) => `
                    <span class="score-badge ${badge.css}">
                        ${badge.label}: ${badge.value.toFixed(1)}/100
                    </span>
                `,
            )
            .join("") || '<span class="score-badge neutral">Scores appear after face detection</span>';
    }
    if (summarySelectors.confidence_bar) {
        summarySelectors.confidence_bar.style.width = `${hasFace ? confidencePercent : 0}%`;
    }
    if (summarySelectors.face_recommendation) {
        summarySelectors.face_recommendation.textContent =
            detail.recommendation ||
            (["basic-fallback", "face-detection-fallback"].includes(detail.analysis_mode)
                ? "A basic fatigue estimate is active. Improve lighting and face the camera for detailed eye cues."
                : "Adjust lighting and keep each face fully in frame so the analysis can read the eyes, posture, and fatigue risk.");
    }
}

function clearCameraUnavailable() {
    if (cameraFrame) {
        cameraFrame.classList.remove("camera-error");
    }
    if (cameraFeed) {
        cameraFeed.style.display = "block";
    }
    if (cameraAccessBanner) {
        cameraAccessBanner.innerHTML = `
            <div class="camera-access-icon">📷</div>
            <div>
                <strong>Allow camera access</strong>
                <p>Use localhost or HTTPS in the browser, then click Start camera and approve the webcam permission request.</p>
            </div>
        `;
    }
}

function getCameraConstraints(preferredDevice = "default") {
    const baseConstraints = {
        width: { ideal: 1280 },
        height: { ideal: 720 },
    };

    if (preferredDevice === "rear") {
        return { ...baseConstraints, facingMode: { ideal: "environment" } };
    }

    if (preferredDevice === "front") {
        return { ...baseConstraints, facingMode: { ideal: "user" } };
    }

    if (preferredDevice && preferredDevice !== "default" && !preferredDevice.startsWith("default")) {
        return {
            ...baseConstraints,
            deviceId: { ideal: preferredDevice },
        };
    }

    return {
        ...baseConstraints,
        facingMode: { ideal: /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ? "environment" : "user" },
    };
}

async function toggleCamera() {
    if (!cameraFeed) {
        return;
    }

    if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
        localStream = null;
        cameraFeed.srcObject = null;
        cameraFeed.classList.remove("camera-mirrored");
        if (cameraStatus) {
            cameraStatus.textContent = "Camera off";
            cameraStatus.className = "status-pill idle";
        }
        if (cameraToggle) {
            cameraToggle.textContent = "Start camera";
        }
        if (cameraContext) {
            cameraContext.textContent = "Camera paused. Ready for the next face scan.";
        }
        clearCameraUnavailable();
        return;
    }

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Browser camera API is not supported.");
        }

        const isSecure = window.isSecureContext || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
        if (!isSecure) {
            throw new Error("Camera access requires a secure context. Please use localhost or HTTPS.");
        }

        const preferredDevice = dashboardConfig.controls.camera_source || "default";
        const videoConstraints = getCameraConstraints(preferredDevice);

        localStream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false,
        });

        clearCameraUnavailable();
        cameraFeed.srcObject = localStream;
        const actualFacing = localStream.getVideoTracks()[0]?.getSettings?.().facingMode;
        const requestedFacing = dashboardConfig.controls.camera_source || "default";
        const mirrorPreview = actualFacing ? actualFacing === "user" : requestedFacing === "front" ||
            (requestedFacing === "default" && !/Android|iPhone|iPad|iPod/i.test(navigator.userAgent));
        cameraFeed.classList.toggle("camera-mirrored", mirrorPreview);
        await cameraFeed.play().catch(() => undefined);

        if (cameraStatus) {
            cameraStatus.textContent = "Camera live";
            cameraStatus.className = "status-pill live";
        }
        if (cameraToggle) {
            cameraToggle.textContent = "Stop camera";
        }
        if (cameraContext) {
            cameraContext.textContent = `${analysisModeSelect?.value || dashboardConfig.default_mode} live feed · face fatigue analysis active`;
        }
    } catch (error) {
        console.error(error);
        setCameraUnavailable(
            error.message.includes("secure context")
                ? "Use localhost in the browser and click Start camera again."
                : "Camera blocked. Please allow access in the browser and try again.",
        );
    }
}

function setCapturedPhoto(assetDataUrl) {
    if (!photoPanel || !capturedPhoto) {
        return;
    }
    capturedPhoto.src = assetDataUrl;
    photoPanel.style.display = "block";
}

function clearCapturedPhoto() {
    if (capturedPhoto) {
        capturedPhoto.removeAttribute("src");
    }
    if (photoPanel) {
        photoPanel.style.display = "none";
    }
}

async function analyzeImageData(imageData, label = "camera") {
    try {
        const studentName = document.getElementById("student-name")?.value.trim() || "";
        const studentClass = document.getElementById("student-class")?.value.trim() || "";
        const studentRoll = document.getElementById("student-roll")?.value.trim() || "";
        const saveStudentScan = document.getElementById("save-student-scan")?.checked === true;
        if (saveStudentScan && (!studentName || !studentClass || !studentRoll)) {
            throw new Error("Enter the student's name, class, and roll number to save this scan.");
        }
        const response = await fetch("/api/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image_data: imageData,
                source: label,
                ...(saveStudentScan ? { student: { name: studentName, class_name: studentClass, roll_number: studentRoll } } : {}),
                detection_mode: dashboardConfig.controls.student_tracking === false ? "single-user" : (dashboardConfig.controls.detection_mode || "multi-face"),
                detection_settings: {
                    ear_blink_cutoff: Number(dashboardConfig.controls.ear_blink_cutoff ?? 0.18),
                    mar_yawn_sensitivity: Number(dashboardConfig.controls.mar_yawn_sensitivity ?? 0.24),
                    posture_tilt_degrees: Number(dashboardConfig.controls.posture_tilt_degrees ?? 18),
                    posture_analysis: dashboardConfig.controls.posture_analysis !== false,
                    fatigue_detection: dashboardConfig.controls.fatigue_detection !== false,
                    phone_detection: dashboardConfig.controls.phone_detection !== false,
                    phone_confidence: Number(dashboardConfig.controls.phone_confidence ?? 0.12),
                },
            }),
        });

        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            const scanError = new Error(result.error || `Face scan failed (HTTP ${response.status})`);
            scanError.scanFailure = true;
            throw scanError;
        }
        const attention = Number(result.attention || 0);
        const fatigue = Number(result.fatigue || 0);
        renderFaceDetails(result);
        renderPhoneDetections(result);

        if (summarySelectors.average_attention) {
            summarySelectors.average_attention.textContent = `${attention.toFixed(1)}%`;
        }
        if (summarySelectors.eye_engagement) {
            summarySelectors.eye_engagement.textContent = `${(Number(result.eye_engagement || 0) * 100).toFixed(1)}%`;
        }
        const faceMetrics = Array.isArray(result.faces) && result.faces.length ? result.faces : (result.face_detected ? [result] : []);
        const alertThreshold = Number(dashboardConfig.controls.alert_threshold ?? 0.7);
        const fatigueAlertCount = dashboardConfig.controls.fatigue_detection === false ? 0 : faceMetrics.filter((face) =>
            Number(face.fatigue_index ?? Number(face.fatigue || 0) * 100) / 100 >= alertThreshold ||
            ["Drowsy", "Yawning"].includes(face.drowsiness_state),
        ).length;
        if (summarySelectors.fatigue_alerts) summarySelectors.fatigue_alerts.textContent = String(fatigueAlertCount);
        if (summarySelectors.active_students) summarySelectors.active_students.textContent = String(faceMetrics.length);
        if (summarySelectors.student_count) summarySelectors.student_count.textContent = `${faceMetrics.length} detected`;
        if (summarySelectors.posture_score) {
            summarySelectors.posture_score.textContent = `${(Number(result.posture_score || 0) * 100).toFixed(1)}%`;
        }
        if (summarySelectors.last_refresh) {
            summarySelectors.last_refresh.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }

        if (cameraStatus) {
            if (result.phone_detections?.length) {
                cameraStatus.textContent = `PHONE IS DETECTED · ${result.phone_detections.length} found`;
                cameraStatus.className = "status-pill idle";
            } else if (!result.face_detected) {
                cameraStatus.textContent = "No face detected · retry scan";
                cameraStatus.className = "status-pill idle";
            } else {
                cameraStatus.textContent = `${result.face_count || faceMetrics.length} face${(result.face_count || faceMetrics.length) === 1 ? "" : "s"} analyzed`;
                cameraStatus.className = "status-pill live";
            }
        }
        if (summarySelectors.room_status && summarySelectors.status_detail) {
            if (!result.face_detected) {
                summarySelectors.room_status.textContent = "Waiting";
                summarySelectors.status_detail.textContent = "No clear face in this frame";
            } else if (attention >= 80) {
                summarySelectors.room_status.textContent = "Strong";
                summarySelectors.status_detail.textContent = "Face active and focused";
            } else if (attention >= 60) {
                summarySelectors.room_status.textContent = "Stable";
                summarySelectors.status_detail.textContent = "Some fatigue risk";
            } else {
                summarySelectors.room_status.textContent = "Watch";
                summarySelectors.status_detail.textContent = "Needs attention";
            }
        }

        if (result.saved_scan) await loadStudentScanHistory();
        return result;
    } catch (error) {
        console.error(error);
        if (cameraStatus) {
            cameraStatus.textContent = error.message || "Scan unavailable";
            cameraStatus.className = "status-pill idle";
        }
        if (error.scanFailure || error instanceof TypeError) {
            if (summarySelectors.face_status) summarySelectors.face_status.textContent = "Scan failed";
            if (summarySelectors.face_risk) summarySelectors.face_risk.textContent = "Not evaluated";
            if (summarySelectors.face_attention) summarySelectors.face_attention.textContent = "--";
            if (summarySelectors.face_eye) summarySelectors.face_eye.textContent = "--";
            if (summarySelectors.face_posture) summarySelectors.face_posture.textContent = "--";
            if (summarySelectors.face_fatigue) summarySelectors.face_fatigue.textContent = "--";
            if (summarySelectors.face_confidence) summarySelectors.face_confidence.textContent = "--";
            if (summarySelectors.overall_score_badge) summarySelectors.overall_score_badge.textContent = "--";
            if (summarySelectors.score_badges) summarySelectors.score_badges.textContent = "No scores: scan failed";
            if (summarySelectors.drowsiness_status_badge) summarySelectors.drowsiness_status_badge.textContent = "Not evaluated";
            if (summarySelectors.confidence_bar) summarySelectors.confidence_bar.style.width = "0%";
            if (summarySelectors.face_recommendation) summarySelectors.face_recommendation.textContent = error.message || "The server could not analyze this frame. Check server logs and try again.";
            if (summarySelectors.average_attention) summarySelectors.average_attention.textContent = "--";
            if (summarySelectors.eye_engagement) summarySelectors.eye_engagement.textContent = "--";
            if (summarySelectors.posture_score) summarySelectors.posture_score.textContent = "--";
            if (summarySelectors.active_students) summarySelectors.active_students.textContent = "--";
            if (summarySelectors.student_count) summarySelectors.student_count.textContent = "Scan failed";
            if (summarySelectors.room_status) summarySelectors.room_status.textContent = "Error";
            if (summarySelectors.status_detail) summarySelectors.status_detail.textContent = "No scores were produced";
            if (summarySelectors.fatigue_alerts) summarySelectors.fatigue_alerts.textContent = "--";
            if (phoneDetectionStatus) {
                phoneDetectionStatus.classList.remove("detected");
                phoneDetectionStatus.classList.add("unavailable");
                phoneDetectionStatus.textContent = "Phone scan skipped because image analysis failed";
            }
            if (phoneOverlay) phoneOverlay.getContext("2d")?.clearRect(0, 0, phoneOverlay.width, phoneOverlay.height);
        }
        throw error;
    }
}

async function capturePhotoForAnalysis() {
    if (capturePhotoButton?.disabled) return;
    if (!cameraFeed || !cameraFeed.videoWidth || !localStream) {
        if (cameraStatus) {
            cameraStatus.textContent = "Start camera first";
            cameraStatus.className = "status-pill idle";
        }
        return;
    }

    try {
        const saveScan = document.getElementById("save-student-scan")?.checked === true;
        const identity = ["student-name", "student-class", "student-roll"]
            .map((id) => document.getElementById(id)?.value.trim() || "");
        if (saveScan && identity.some((value) => !value)) {
            if (cameraStatus) {
                cameraStatus.textContent = "Enter student name, class, and roll number";
                cameraStatus.className = "status-pill idle";
            }
            return;
        }
        if (scanProgressOverlay) scanProgressOverlay.hidden = false;
        cameraFrame?.setAttribute("aria-busy", "true");
        if (capturePhotoButton) {
            capturePhotoButton.disabled = true;
            capturePhotoButton.dataset.originalText = capturePhotoButton.textContent;
            capturePhotoButton.textContent = "Analyzing photo…";
            capturePhotoButton.setAttribute("aria-busy", "true");
        }
        if (cameraStatus) {
            cameraStatus.textContent = "Analyzing face, fatigue, and phone cues…";
            cameraStatus.className = "status-pill idle";
        }
        const canvas = document.createElement("canvas");
        const captureScale = Math.min(1, 1280 / Math.max(cameraFeed.videoWidth, cameraFeed.videoHeight));
        canvas.width = Math.round(cameraFeed.videoWidth * captureScale);
        canvas.height = Math.round(cameraFeed.videoHeight * captureScale);
        const context = canvas.getContext("2d");
        context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL("image/jpeg", 0.82);
        setCapturedPhoto(imageData);
        const scan = await analyzeImageData(imageData, "student-photo");
        if (scan?.saved_scan && cameraStatus) {
            cameraStatus.textContent = `Saved for ${scan.saved_scan.name}`;
            cameraStatus.className = "status-pill live";
        }
        await refreshDashboard();
    } catch (error) {
        console.error(error);
        if (cameraStatus) {
            cameraStatus.textContent = error.message || "Scan unavailable";
            cameraStatus.className = "status-pill idle";
        }
    } finally {
        if (scanProgressOverlay) scanProgressOverlay.hidden = true;
        cameraFrame?.removeAttribute("aria-busy");
        if (capturePhotoButton) {
            capturePhotoButton.disabled = false;
            capturePhotoButton.textContent = capturePhotoButton.dataset.originalText || "Capture & analyze photo";
            capturePhotoButton.removeAttribute("aria-busy");
        }
    }
}

function updateCameraFacingButtons() {
    const facing = dashboardConfig.controls.camera_source || "default";
    cameraFacingButtons.forEach((button) => {
        const selected = button.dataset.facingCamera === facing;
        button.setAttribute("aria-pressed", String(selected));
        button.classList.toggle("active", selected);
    });
}

cameraFacingButtons.forEach((button) => {
    button.addEventListener("click", async () => {
        const facing = button.dataset.facingCamera;
        if (!facing || dashboardConfig.controls.camera_source === facing) return;
        dashboardConfig.controls.camera_source = facing;
        if (cameraSourceSelect) cameraSourceSelect.value = facing;
        saveConfigToStorage();
        updateCameraFacingButtons();
        if (localStream) {
            await toggleCamera();
            await toggleCamera();
        }
    });
});
updateCameraFacingButtons();

if (analysisModeSelect) {
    analysisModeSelect.addEventListener("change", (event) => {
        dashboardConfig.default_mode = event.target.value;
        dashboardConfig.controls.analysis_mode = event.target.value;
        saveConfigToStorage();
        if (cameraContext) {
            cameraContext.textContent = `${event.target.value} feed`;
        }
    });
}

const focusModeSelect = document.getElementById("focus-mode");
if (focusModeSelect) {
    focusModeSelect.addEventListener("change", (event) => {
        dashboardConfig.controls.focus_mode = event.target.value;
        saveConfigToStorage();
        if (summarySelectors.recommended_action) {
            const choice = event.target.value;
            const detailText = {
                lecture: "Keep the current teaching rhythm. Monitor the back row and continue live fatigue checks.",
                exam: "Keep the room quiet and watch signs of stress or distraction across the front row.",
                group: "Watch for discussion drift and quickly rotate attention back to the active group.",
                lab: "Keep students engaged with short check-ins and watch for fatigue during hands-on tasks.",
            };
            summarySelectors.recommended_action.textContent = detailText[choice] || detailText.lecture;
        }
    });
}

if (cameraSourceSelect) {
    cameraSourceSelect.addEventListener("change", async (event) => {
        dashboardConfig.controls.camera_source = event.target.value;
        saveConfigToStorage();
        updateCameraFacingButtons();
        if (localStream) {
            await toggleCamera();
            await toggleCamera();
        }
    });
}

async function loadStudentScanHistory() {
    const target = document.getElementById("student-scan-history-list");
    if (!target) return;
    const params = new URLSearchParams();
    const className = document.getElementById("scan-filter-class")?.value.trim();
    const roll = document.getElementById("scan-filter-roll")?.value.trim();
    if (className) params.set("class", className);
    if (roll) params.set("roll", roll);
    try {
        const response = await fetch(`/api/student-scans?${params}`);
        if (!response.ok) throw new Error("Unable to load saved scans");
        const { scans = [] } = await response.json();
        if (!scans.length) {
            target.innerHTML = '<p class="empty-state">No saved scans match these filters.</p>';
            return;
        }
        target.innerHTML = scans.map((scan) => `
            <article class="student-scan-card">
                <img src="/api/student-scans/${scan.id}/image" alt="Saved scan for ${escapeHtml(scan.name)}" loading="lazy" />
                <div><strong>${escapeHtml(scan.name)}</strong><span>Class ${escapeHtml(scan.class_name)} · Roll ${escapeHtml(scan.roll_number)}</span>
                    <span>${new Date(scan.timestamp).toLocaleString()}</span></div>
                <div class="scan-comparison"><span>Fatigue <strong>${(Number(scan.fatigue) * 100).toFixed(1)}%</strong></span>
                    <span>Attention <strong>${Number(scan.attention).toFixed(1)}%</strong></span>
                    <span class="fatigue-level ${scan.fatigue_level.toLowerCase()}">${escapeHtml(scan.fatigue_level)} fatigue</span></div>
            </article>`).join("");
    } catch (error) {
        target.innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
    }
}

async function loadClassroomComparison() {
    const target = document.getElementById("classroom-student-comparison");
    if (!target) return;
    try {
        const params = new URLSearchParams();
        const className = document.getElementById("classroom-filter-class")?.value.trim();
        const roll = document.getElementById("classroom-filter-roll")?.value.trim();
        if (className) params.set("class", className);
        if (roll) params.set("roll", roll);
        const response = await fetch(`/api/student-scans?${params}`);
        if (!response.ok) throw new Error("Sign in to view student comparisons.");
        const { scans = [] } = await response.json();
        const latestByStudent = new Map();
        for (const scan of scans) {
            const key = `${scan.class_name}:${scan.roll_number}`;
            if (!latestByStudent.has(key)) latestByStudent.set(key, scan);
        }
        const latest = [...latestByStudent.values()];
        if (!latest.length) {
            target.innerHTML = '<p class="empty-state">No saved student scans to compare yet. Save a scan from Overview.</p>';
            return;
        }
        target.innerHTML = latest.map((scan) => `
            <article class="student-scan-card compact-scan-card">
                <div><strong>${escapeHtml(scan.name)}</strong><span>Class ${escapeHtml(scan.class_name)} · Roll ${escapeHtml(scan.roll_number)}</span></div>
                <div class="scan-comparison"><span>Fatigue <strong>${(Number(scan.fatigue) * 100).toFixed(1)}%</strong></span><span>Attention <strong>${Number(scan.attention).toFixed(1)}%</strong></span><span class="fatigue-level ${scan.fatigue_level.toLowerCase()}">${escapeHtml(scan.fatigue_level)}</span></div>
            </article>`).join("");
    } catch (error) {
        target.innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
    }
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}

for (const id of ["refresh-student-scans", "scan-filter-class", "scan-filter-roll"]) {
    document.getElementById(id)?.addEventListener(id === "refresh-student-scans" ? "click" : "change", loadStudentScanHistory);
}
for (const id of ["classroom-filter-class", "classroom-filter-roll"]) {
    document.getElementById(id)?.addEventListener("change", loadClassroomComparison);
}
loadStudentScanHistory();
loadClassroomComparison();

if (detectionModeSelect) {
    detectionModeSelect.addEventListener("change", (event) => {
        dashboardConfig.controls.detection_mode = event.target.value;
        saveConfigToStorage();
    });
}

if (earBlinkCutoffInput) {
    earBlinkCutoffInput.addEventListener("input", (event) => {
        const value = Number(event.target.value);
        dashboardConfig.controls.ear_blink_cutoff = value;
        if (earBlinkCutoffLabel) {
            earBlinkCutoffLabel.textContent = value.toFixed(2);
        }
        saveConfigToStorage();
    });
}

if (marYawnSensitivityInput) {
    marYawnSensitivityInput.addEventListener("input", (event) => {
        const value = Number(event.target.value);
        dashboardConfig.controls.mar_yawn_sensitivity = value;
        if (marYawnSensitivityLabel) {
            marYawnSensitivityLabel.textContent = value.toFixed(2);
        }
        saveConfigToStorage();
    });
}

if (postureTiltDegreesInput) {
    postureTiltDegreesInput.addEventListener("input", (event) => {
        const value = Number(event.target.value);
        dashboardConfig.controls.posture_tilt_degrees = value;
        if (postureTiltDegreesLabel) {
            postureTiltDegreesLabel.textContent = `${value.toFixed(0)}°`;
        }
        saveConfigToStorage();
    });
}

if (alertThresholdInput) {
    alertThresholdInput.addEventListener("input", (event) => {
        const threshold = Number(event.target.value);
        dashboardConfig.controls.alert_threshold = threshold;
        if (alertThresholdLabel) {
            alertThresholdLabel.textContent = threshold.toFixed(2);
        }
        saveConfigToStorage();
    });
}

if (phoneConfidenceInput) {
    phoneConfidenceInput.addEventListener("input", (event) => {
        const value = Number(event.target.value);
        dashboardConfig.controls.phone_confidence = value;
        if (phoneConfidenceLabel) phoneConfidenceLabel.textContent = value.toFixed(2);
        saveConfigToStorage();
    });
}

if (optionControls) {
    optionControls.addEventListener("change", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLInputElement)) {
            return;
        }
        const optionKey = target.dataset.option;
        if (!optionKey) {
            return;
        }
        dashboardConfig.controls[optionKey] = target.checked;
        saveConfigToStorage();
        updateStatusSummary();
        if (summarySelectors.hero_snapshot) {
            summarySelectors.hero_snapshot.textContent = target.checked ? "Active" : "Paused";
        }
        if (optionKey === "camera_feed" && cameraStatus) {
            cameraStatus.textContent = target.checked ? "Camera ready" : "Camera off";
            cameraStatus.className = "status-pill idle";
        }
        applyCameraOptions();
        if (optionKey === "auto_refresh") configureLiveTimers();
    });
}

const saveSettingsButton = document.getElementById("save-settings");
if (saveSettingsButton) {
    saveSettingsButton.addEventListener("click", () => {
        saveConfigToStorage();
        updateStatusSummary();
        if (cameraStatus) {
            cameraStatus.textContent = "Settings saved";
            cameraStatus.className = "status-pill live";
            setTimeout(() => {
                if (localStream) {
                    cameraStatus.textContent = "Camera live";
                    cameraStatus.className = "status-pill live";
                } else {
                    cameraStatus.textContent = "Camera off";
                    cameraStatus.className = "status-pill idle";
                }
            }, 1200);
        }
    });
}

if (cameraToggle) {
    cameraToggle.addEventListener("click", toggleCamera);
}
if (document.getElementById("privacy-wireframe-toggle")) {
    document.getElementById("privacy-wireframe-toggle").addEventListener("click", togglePrivacyWireframe);
}
if (capturePhotoButton) {
    capturePhotoButton.addEventListener("click", capturePhotoForAnalysis);
}
if (clearPhotoButton) {
    clearPhotoButton.addEventListener("click", () => {
        clearCapturedPhoto();
        if (cameraStatus) {
            cameraStatus.textContent = localStream ? "Camera live" : "Camera off";
            cameraStatus.className = localStream ? "status-pill live" : "status-pill idle";
        }
    });
}
if (syncDashboardButton) {
    syncDashboardButton.addEventListener("click", refreshDashboard);
}
if (logoutButton) {
    logoutButton.addEventListener("click", async () => {
        await fetch("/api/auth/logout", { method: "POST" });
        window.location.href = "/";
    });
}

const roomStatusDisplay = document.getElementById("room-status-display");
if (roomStatusDisplay && summarySelectors.room_status) {
    roomStatusDisplay.textContent = summarySelectors.room_status.textContent || "Waiting for scan";
    const roomStatusObserver = new MutationObserver(() => {
        roomStatusDisplay.textContent = summarySelectors.room_status.textContent || "Waiting for scan";
    });
    roomStatusObserver.observe(summarySelectors.room_status, { childList: true, characterData: true, subtree: true });
}

loadConfigFromStorage();
renderControls();
loadCameraDevices();
loadDashboardConfig();
refreshDashboard();
configureLiveTimers();

if (cameraToggle && !window.isSecureContext && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    setCameraUnavailable("Use localhost or HTTPS, then click Start camera.");
}
