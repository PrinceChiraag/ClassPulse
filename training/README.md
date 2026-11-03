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
