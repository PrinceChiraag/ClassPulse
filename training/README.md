# Phone model training tools

Run these commands from the repository root with its `.venv` activated:

```powershell
python training\validate_phone_dataset.py
python training\train_phone_detector.py
python training\evaluate_phone_detector.py
python training\test_phone_detector.py .\path\to\image-or-video.mp4
```

Training always uses CPU, validates the dataset first, and writes the best
checkpoint to `models/phone_best.pt`. It is never run automatically.

At runtime ClassPulse loads this custom one-class phone detector by default,
resolved relative to the repository directory (or `/app` in the Docker image).
Set `CLASSPULSE_PHONE_MODEL` to select another compatible Ultralytics model.
The model must include a class named `phone`, `smartphone`, `cell phone`,
`mobile phone`, `handset`, or `cellular phone` for phone alerts. General object
Phone scans use the custom detector, while general object scans also use the
bundled `yolo11n.pt` COCO model. The app runs both models on CPU; detector
availability and failures are reported by `/health` and the application logs.
