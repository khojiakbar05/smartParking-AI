"""Capture representative selected-camera frames and compare YOLO on CPU/MPS."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

from camera_devices import open_selected_camera


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck", "van", "motor"}


def detection_rows(result, names):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coords = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    rows = []
    for box, score, class_id in zip(coords, scores, class_ids):
        if isinstance(names, dict):
            name = names.get(int(class_id), str(class_id))
        else:
            name = names[int(class_id)]
        rows.append({
            "class_id": int(class_id),
            "name": str(name),
            "confidence": float(score),
            "box": [float(value) for value in box],
        })
    return rows


def signatures_match(first, second, confidence_tolerance=0.001,
                     box_tolerance=1.0):
    """Compare corresponding detections within cross-device numeric tolerance."""
    sort_key = lambda row: (row["class_id"], *row["box"])
    first = sorted(first, key=sort_key)
    second = sorted(second, key=sort_key)
    if len(first) != len(second):
        return False
    for left, right in zip(first, second):
        if left["class_id"] != right["class_id"]:
            return False
        if abs(left["confidence"] - right["confidence"]) > confidence_tolerance:
            return False
        if any(abs(a - b) > box_tolerance
               for a, b in zip(left["box"], right["box"])):
            return False
    return True


def print_rows(title, rows):
    print(f"{title}: {len(rows)} raw detections")
    for row in rows:
        coords = [round(value, 1) for value in row["box"]]
        print(
            f"  class_id={row['class_id']} name={row['name']} "
            f"confidence={row['confidence']:.3f} bbox={coords}"
        )
    vehicles = sum(row["name"] in VEHICLE_LABELS for row in rows)
    print(f"  vehicle detections (model's vehicle labels): {vehicles}")


def capture_frames(count, interval_seconds, output_dir):
    capture = None
    frames = []
    try:
        capture = open_selected_camera()
        print("Camera opened with saved AVFoundation selection.")
        next_capture_time = 0.0
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Camera frame read failed at sample {len(frames) + 1}/{count}."
                )
            now = time.monotonic()
            if now < next_capture_time:
                time.sleep(min(0.05, next_capture_time - now))
                continue
            if frame.dtype != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(
                    f"Invalid camera frame: shape={frame.shape}, dtype={frame.dtype}"
                )
            index = len(frames) + 1
            path = output_dir / f"frame_{index:02d}.jpg"
            if not cv2.imwrite(
                str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Could not save camera frame: {path}")
            saved = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if saved is None:
                raise RuntimeError(f"Could not re-open saved image: {path}")
            frames.append((path, saved))
            print(
                f"Captured {path}: {saved.shape[1]}x{saved.shape[0]} "
                f"dtype={saved.dtype} channels=BGR"
            )
            next_capture_time = now + interval_seconds
    finally:
        if capture is not None:
            capture.release()
            print("Camera released before model inference.")
    return frames


def diagnose(frames, model, model_label, image_size, confidence, devices):
    comparisons = []
    report = []
    for frame_index, (image_path, image) in enumerate(frames, start=1):
        print(f"\n=== Frame {frame_index}: {image_path} ===")
        results_by_device = {}
        for device in devices:
            try:
                result = model.predict(
                    source=image,
                    imgsz=image_size,
                    conf=confidence,
                    classes=None,
                    device=device,
                    verbose=False,
                )[0]
            except Exception as exc:
                print(f"{device} inference failed: {type(exc).__name__}: {exc}")
                continue

            rows = detection_rows(result, model.names)
            results_by_device[device] = rows
            print_rows(
                f"{device.upper()} (imgsz={image_size}, conf={confidence:.2f})",
                rows,
            )
            annotated_path = image_path.with_name(
                f"{image_path.stem}_{model_label}_{device}_annotated.jpg"
            )
            if not cv2.imwrite(
                str(annotated_path),
                result.plot(labels=True, conf=True, boxes=True),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                print(f"Could not save annotation: {annotated_path}")
            else:
                print(f"Annotated image: {annotated_path}")

        comparison_match = None
        if "cpu" in results_by_device and "mps" in results_by_device:
            comparison_match = signatures_match(
                results_by_device["cpu"], results_by_device["mps"]
            )
            comparisons.append(comparison_match)
            print(
                "CPU/MPS detections match within 1px boxes / 0.001 confidence: "
                f"{comparison_match}"
            )
        elif len(devices) > 1:
            comparisons.append(None)
        report.append({
            "frame": str(image_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "imgsz": image_size,
            "confidence_threshold": confidence,
            "detections": results_by_device,
            "cpu_mps_signature_match": comparison_match,
        })
    if comparisons:
        completed = [item for item in comparisons if item is not None]
        print(
            "\nCPU/MPS comparison summary: "
            f"{sum(completed)}/{len(completed)} exact matches; "
            f"{len(comparisons) - len(completed)} incomplete comparisons."
        )
    return report


def load_frames(input_dir):
    """Load only the original frame files, excluding generated annotations."""
    frame_paths = sorted(input_dir.glob("frame_[0-9][0-9].jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frame_NN.jpg images found in {input_dir}")
    frames = []
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read saved image: {path}")
        frames.append((path, image))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Re-run inference on saved frame_NN.jpg images without opening a camera.",
    )
    args = parser.parse_args()
    if args.count < 1 or args.interval < 0 or args.imgsz < 32:
        parser.error("count/imgsz must be positive and interval cannot be negative")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("conf must be between 0 and 1")
    checkpoint = args.model.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")

    if args.input_dir is not None:
        output_dir = args.output_dir or args.input_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = load_frames(args.input_dir)
        print(f"Loaded {len(frames)} saved frames from {args.input_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or BASE_DIR / "debug" / f"camera_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        frames = capture_frames(args.count, args.interval, output_dir)

    print(f"\nLoading checkpoint once: {checkpoint}")
    model = YOLO(str(checkpoint))
    expected = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 67: "cell phone"}
    print(
        f"task={model.task} nc={getattr(model.model, 'nc', None)} "
        f"names_count={len(model.names)} checkpoint_bytes={checkpoint.stat().st_size}"
    )
    if isinstance(model.names, dict):
        print("Class names=" + ", ".join(
            f"{class_id}:{name}" for class_id, name in model.names.items()
        ))
    else:
        print(f"Class names={model.names}")
    if checkpoint == MODEL_PATH.resolve():
        mismatches = {
            class_id: (expected_name, model.names.get(class_id))
            for class_id, expected_name in expected.items()
            if model.names.get(class_id) != expected_name
        }
        if model.task != "detect" or model.model.nc != 80 or mismatches:
            raise RuntimeError(f"Checkpoint/COCO metadata mismatch: {mismatches}")
    elif model.task != "detect":
        raise RuntimeError(f"Expected detection checkpoint, got task={model.task}")

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    else:
        print("MPS is unavailable in this Python process; CPU only.")
    print(
        f"Inference settings: devices={devices}, imgsz={args.imgsz}, "
        f"conf={args.conf:.2f}, classes=None (unfiltered)"
    )
    model_label = checkpoint.stem.replace(" ", "_")
    report = diagnose(frames, model, model_label, args.imgsz, args.conf, devices)
    report_path = output_dir / f"inference_report_{model_label}.json"
    report_path.write_text(
        json.dumps({
            "checkpoint": str(checkpoint),
            "task": model.task,
            "class_count": model.model.nc,
            "class_names": {str(k): v for k, v in model.names.items()}
            if isinstance(model.names, dict) else list(model.names),
            "devices": devices,
            "frames": report,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Machine-readable inference report: {report_path}")
    print(f"\nAll debug images: {output_dir}")


if __name__ == "__main__":
    main()
