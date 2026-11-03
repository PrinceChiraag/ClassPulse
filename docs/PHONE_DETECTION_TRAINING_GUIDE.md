# ClassPulse phone detection training guide

## Current setup and dataset status

The runtime detector is [`classpulse/phone_detection.py`](../classpulse/phone_detection.py). It now prefers `models/phone_best.pt` when that file exists and falls back to the bundled `yolo11n.pt`, whose COCO class 67 is `cell phone`. Phone detection remains a separate optional pass and does not change fatigue scores. The legacy [`datasets/phone`](../datasets/phone) folder has only an example YAML; the new workspace is `datasets/phone_detection/`.

No dataset images or labels were downloaded or supplied in this repository. The new train, validation, and test folders currently contain only `.gitkeep` placeholders, so validation will fail and training will refuse to start. You must add and review data before training. Do not place identifiable classroom images in this repository unless you have permission and an approved retention plan.

## Candidate public source and license

Roboflow Universe's [mobilephone object detection dataset](https://universe.roboflow.com/m17865515473-163-com/mobilephone-wusj2) lists a **CC BY 4.0** license, one mobile-phone class, and dataset version 5 with 4,951 images. The project overview also reports 3,523 images, so inspect the exact export/version you choose and record its actual counts. The license requires attribution and permits reuse/adaptation, including commercial use, subject to its terms. Check the dataset page and the [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) again before use; public visibility is not a substitute for reviewing image provenance and permissions.

The page exposes download/export actions and API snippets that use an API key. No dataset was downloaded here. To avoid sharing a private key in chat or source control, sign into Roboflow in your browser, select the exact dataset version, export **YOLOv8** format, and download the ZIP. Extract locally and merge only after verifying labels and image rights. If using an API key, store it in a local environment variable or an untracked `.env`; never commit it. The selected dataset is general mobile-phone imagery, not guaranteed to contain enough classroom desk, occlusion, or negative examples, so collect permitted classroom-like images yourself and annotate them.

## Folder layout and annotations

Place files in the matching split directories:

```text
datasets/phone_detection/
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
```

Keep images from the same video/session together in one split to avoid leakage. A reasonable starting split is 70% train, 20% validation, and 10% test, with positive and negative examples in every split. Do not use adjacent video frames across splits. Include real phones in hands, on desks, partly occluded, at varied distances/orientations and lighting; include negatives such as books, calculators, notebooks, and empty scenes. Only annotate phones that are actually visible.

Use an annotation tool such as CVAT or LabelImg and export YOLO detection labels. One object per line:

```text
class_id x_center y_center width height
```

Coordinates are normalized from 0 to 1. This dataset uses class `0` named `phone`. Example:

```text
0 0.512 0.438 0.120 0.250
```

Every image needs a same-stem `.txt` file; use an empty file for a negative image. Do not fabricate labels. The validator checks readability, matching files, class IDs, normalized values, positive box sizes, empty negative labels, split content, and exact duplicate image files across splits.

## Validate

From PowerShell at the project root, with `.venv` activated:

```powershell
python training\validate_phone_dataset.py
```

Correct every reported issue before training. The default requires at least one positive and one negative image across all splits; for meaningful evaluation, collect substantially more balanced and varied examples. The validator does not determine whether human-drawn boxes are semantically accurate or detect visually near-duplicate frames; review labels and keep source videos grouped when making splits.

## CPU training

Training uses Ultralytics, CPU only, image size 640, batch 2, 50 epochs, two workers, and patience 10 by default. It checks available RAM and will stop if less than 4 GiB are free. Override parameters as needed:

```powershell
python training\train_phone_detector.py
python training\train_phone_detector.py --epochs 30 --imgsz 512 --batch 1 --workers 1 --patience 8
```

Training is never started automatically. It starts only when you run the command. The base `yolo11n.pt` is read as pretrained weights and never overwritten. The Ultralytics run is saved beneath `runs/phone-detection/`; the selected best weights are copied to `models/phone_best.pt`.

## Evaluate and test

Once the dataset has a nonempty test split and training has produced weights:

```powershell
python training\evaluate_phone_detector.py --imgsz 640 --batch 2
python training\test_phone_detector.py .\path\to\image.jpg --confidence 0.25 --imgsz 640
python training\test_phone_detector.py .\path\to\video.mp4 --confidence 0.25 --imgsz 640
```

Evaluation prints precision, recall, mAP50, and mAP50-95. Image/video predictions include per-box confidence and save annotated output below `runs/phone-predictions/`. Results depend on dataset quality and must not be treated as 100% accurate. A box is an uncertain object cue, not proof of phone use, distraction, or fatigue.

## Integrate and deploy

Local runs automatically use `models/phone_best.pt` when present, with the existing COCO `yolo11n.pt` as fallback. `CLASSPULSE_PHONE_MODEL` can point to a different local checkpoint; a missing configured path logs a warning and falls back to the bundled model. The detector model is loaded once per process and inference is serialized. Existing dashboard behavior and fatigue calculations remain separate.

For a deployment that should use the trained checkpoint, provide it through a protected artifact or mounted volume at `/app/models/phone_best.pt`, or set `CLASSPULSE_PHONE_MODEL` to its mounted path. Do not add weights or private data to the Docker build context. The current Render-style Docker image keeps the bundled `yolo11n.pt`; if the custom artifact is absent, it will fall back to it. Keep model artifacts and raw datasets out of git.

## Images still to collect

Before relying on classroom performance, collect or source permitted examples of phones held naturally, on desks, edge-on and face-on, partly covered by hands/books, small/far from camera, across varied devices and lighting. Collect hard negatives (calculators, remotes, notebooks, wallets, dark rectangular objects, empty desks). Obtain any required consent, avoid student-identifiable frames when possible, and group adjacent frames from each recording in the same split. Record provenance, license, source-session ID, and annotation review status outside private image data.
