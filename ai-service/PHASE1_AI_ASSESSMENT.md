# SmartParking AI: architecture and Phase 1 assessment

## Scope and architecture

The repository contains a Python AI service, not a React application. Its live path is:

1. `camera_devices.py` opens the saved macOS AVFoundation camera selection.
2. `parking_detection.py` continuously captures the newest frame, runs the local Ultralytics YOLO11n checkpoint, filters COCO vehicle IDs, compares vehicle boxes with saved slot polygons, smooths occupancy, and draws a preview.
3. `parking_logic.py` validates/scales the saved polygons, calculates box/polygon intersection ratios, and performs a majority vote.
4. `parking_slot_editor.py` is the existing interactive slot editor.

The project currently has no HTTP API module or React/frontend files. FastAPI and Uvicorn are already listed in `requirements.txt`, so the future dashboard seam can be a small API wrapping the existing occupancy engine. A response can expose `slot_id`, `status`, `confidence`, `timestamp`, plus `vehicle_count`, `total_slots`, `available_slots`, and `occupied_slots`. `UNKNOWN` and confidence are not currently part of the occupancy engine; that belongs to Phase 2/3 and is not claimed as implemented here.

## Implementation plan

- **Phase 1 (done):** verify checkpoint metadata and COCO mapping; compare the same image on CPU and MPS; test the selected camera frame and the saved overhead parking image; document model/dataset suitability without downloading weights.
- **Phase 2 (next):** retain the slot polygons and camera capture; add a per-slot image classifier and an explicit `UNKNOWN` result; combine it with vehicle-box overlap only when there is supporting visual evidence; smooth only valid observations.
- **Phase 3 (next):** expose the existing engine through a small FastAPI service with the agreed slot/summary response schema.
- **Phase 4 (next):** connect an existing React dashboard if/when its source is present. No frontend exists in this repository to integrate today.
- **Phase 5 (in progress):** serialize AVFoundation capture reads and release, close camera handles before diagnostic inference, and keep cleanup in `finally`; continue testing preview shutdown on the target Mac.

## Phase 1 findings

### Checkpoint and label mapping

- Checkpoint: `yolo11n.pt`, 5,613,764 bytes, SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Ultralytics reports detection task, 80 classes, `nc=80`.
- Mapping is correct: `2=car`, `3=motorcycle`, `5=bus`, `7=truck`, `67=cell phone` (see [Ultralytics COCO class map](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)).
- Vehicle filtering uses IDs `[2, 3, 5, 7]` in Python after inference. The raw detection path preserves all classes for diagnosis. There is no shifted class map and no evidence of incorrect filtering.

### Image experiments

**Fresh selected-camera capture:** `parking_debug.jpg` was captured through the existing selected AVFoundation camera and re-read through OpenCV before inference. It is a blurry close-up of a desk/object, not a parking-lot view. Raw YOLO results at the 1280 image-size sweep were household-object labels (vase and toothbrush); vehicle count was zero. This explains the zero vehicles for this sampled frame, but does not establish what the user sees in another camera session.

**Saved overhead parking image:** `parking_debug.png` (1415x518) visibly contains parked vehicles. On this exact image, YOLO11n at `imgsz=960`, `conf=0.05`, with no class filter produced car-shaped boxes but labeled them `cell phone` (class 67). CPU and MPS returned identical class/confidence signatures. The visual annotation is saved as `parking_debug_overhead_annotated.jpg`.

The cause of the overhead-image failure is a viewpoint/domain mismatch: standard COCO YOLO11n was trained for general photographic object appearances and does not reliably recognize these top-down car appearances. Lowering confidence adds detections but does not correct their semantics. Do not reinterpret phone detections as vehicles. Retain COCO IDs and use a parking-specific classifier or camera-specific fine-tuning.

### Model and dataset recommendation

For fixed, known spaces, the best first model strategy is **per-slot vacant/occupied classification**, using the existing polygon coordinates to crop each space, then fine-tuning on labeled crops from this exact camera and lighting. PKLot contains 12,417 lot frames and about 695,900 segmented, manually labeled spaces; its authors include top-of-building cameras and vacant/occupied slot crops ([official dataset page](https://web.inf.ufpr.br/vri/databases/parking-lot-database/), [paper](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf)). CNRPark+EXT contains about 150,000 labeled vacant/occupied space patches from 164 spaces and multiple camera views ([official dataset site](https://cnrpark.it/)). Both are directly aligned with slot classification, but camera-specific fine-tuning and validation remain necessary because camera view, crop geometry, lighting, and resolution differ.

For retaining vehicle boxes, fine-tune the existing YOLO architecture on labeled frames from this camera. VisDrone is a possible aerial vehicle pretraining source, with manually annotated vehicle boxes, but it is drone imagery with varied viewpoints rather than a guaranteed match for this fixed overhead camera ([official dataset repository](https://github.com/VisDrone/VisDrone-Dataset)). A small third-party VisDrone checkpoint was subsequently added under `debug/models/` for the follow-up comparison below; it is not used by the production parking application. No training datasets were downloaded.

## Stability notes

AVFoundation capture reads and release are serialized in `LatestFrameCamera`, and diagnostic inference releases the camera handle before running CPU/MPS comparisons. Camera/window cleanup is in `finally` blocks. The faulthandler-enabled camera diagnostic completed and returned to the terminal prompt without a segmentation fault. This is a successful reproduction run, not proof of the original crash's exact cause or a guarantee that the issue cannot recur. The likely failure mechanism addressed is concurrent capture read/release; further long-duration testing on the target Mac is still needed.

## Follow-up: five camera frames and VisDrone model comparison

The new `camera_frame_diagnostics.py` captured five 1920x1080 frames at one-second intervals from the saved AVFoundation selection, saved them under `debug/camera_20261116_035222/`, released the camera, then ran independent CPU and MPS inference on the exact saved JPEG files. It prints every raw class ID, name, confidence, and box and writes per-frame annotations and `inference_report_*.json` files. The frames show a desk/monitor for the first two samples and a parking image displayed on a monitor for the final three; they are not a clean view of a physical parking lot.

At `imgsz=1280`, `conf=0.05`, no class filtering:

| Frame | YOLO11n raw output (MPS) | Vehicle classes | VisDrone YOLOv8n raw output (MPS) |
|---|---|---:|---|
| 1 | 6: toothbrush 3, chair 1, vase 1, toilet 1 | 0 | 1 car (likely false positive on desk/monitor scene) |
| 2 | 1: toilet 1 | 0 | 0 |
| 3 | 15: cell phone 8, cup 6, person 1 | 0 | 10: car 9, van 1 |
| 4 | 24: cup 7, cell phone 7, bottle 8, toaster 1, book 1 | 0 | 16: car 11, van 4, bus 1 |
| 5 | 23: cup 9, cell phone 7, bottle 6, toaster 1 | 0 | 13: car 10, van 3 |

CPU and MPS outputs matched on all five frames for both checkpoints when comparing class IDs, boxes within 1 pixel, and confidences within 0.001. The initial exact-float comparison reported one frame mismatch only because one confidence differed by 0.0001; its labels and boxes were the same. The local MPS path successfully loaded and ran both Ultralytics checkpoints. The VisDrone candidate detects cars on the monitor image where COCO YOLO11n does not, but its first-frame car is a false positive and the screen test does not validate true-lot accuracy.

The comparison checkpoint is the third-party `sobucki/droneflow-yolov8n-visdrone-vehicles` `best.pt`, fetched from pinned revision `42716c7`, size 6,316,394 bytes, SHA-256 `88b25b3cf6c85a824dc5a2751c208d7dee5bb812dc85caa7cd311af7d04ff0a1`. Its metadata maps 0=car, 1=van, 2=truck, 3=bus, 4=motor. The model card reports test-dev mAP50 0.535 at 1280 and describes VisDrone drone traffic footage, not a fixed parking camera; its metrics are self-reported. Its weights/code are AGPL-3.0 and the underlying VisDrone data has academic-use terms, so review licensing before redistribution or product deployment ([model card](https://huggingface.co/sobucki/droneflow-yolov8n-visdrone-vehicles), [Ultralytics VisDrone notes](https://docs.ultralytics.com/datasets/detect/visdrone)). UAVDT is another drone benchmark with about 80,000 annotated frames, but its moving aerial views also differ from a fixed parking view ([UAVDT benchmark](https://sites.google.com/view/daweidu/projects/uavdt)).

For the final fixed-space system, per-slot occupied/free classification remains the stronger architecture choice; the VisDrone detector is a promising vehicle-box candidate for a true-lot comparison, not an occupancy model and not yet a production replacement. No slot coordinates or existing YOLO11n weights were changed.

## Reproduction commands

Run from `ai-service` in the VS Code terminal that has macOS camera permission:

```bash
cd /Users/khojiakbarkarimjonov/Desktop/smartParking-AI/ai-service
PYTHONFAULTHANDLER=1 ./.venv311/bin/python -X faulthandler parking_detection.py --diagnose-yolo
```

This captures `parking_debug.jpg`, runs an unfiltered raw-class diagnostic with confidence thresholds 0.05/0.10/0.15/0.25 and image sizes 640/960/1280, and writes `parking_debug_annotated.jpg`.

Capture five frames and compare the existing YOLO11n model on CPU and MPS:

```bash
PYTHONFAULTHANDLER=1 ./.venv311/bin/python -X faulthandler camera_frame_diagnostics.py
```

Repeat inference on those exact images with YOLO11n at a common 1280 input size:

```bash
PYTHONFAULTHANDLER=1 ./.venv311/bin/python -X faulthandler camera_frame_diagnostics.py --input-dir debug/camera_20261116_035222 --imgsz 1280 --conf 0.05
```

Compare the separately downloaded VisDrone checkpoint on those same images:

```bash
PYTHONFAULTHANDLER=1 ./.venv311/bin/python -X faulthandler camera_frame_diagnostics.py --input-dir debug/camera_20261116_035222 --imgsz 1280 --conf 0.05 --model debug/models/droneflow_yolov8n_visdrone.pt
```

Run the live preview with:

```bash
cd /Users/khojiakbarkarimjonov/Desktop/smartParking-AI/ai-service
PYTHONFAULTHANDLER=1 ./.venv311/bin/python -X faulthandler parking_detection.py
```

Press `Q` to exit. A correct occupancy verification requires the selected camera to show the actual parking lot; the captured `parking_debug.jpg` from this run did not.
