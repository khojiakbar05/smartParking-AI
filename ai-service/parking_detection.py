"""Real-time YOLO11n parking occupancy detection."""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from camera_devices import open_selected_camera
from parking_logic import (
    TemporalOccupancy,
    build_slot_masks,
    load_slots,
    slot_vehicle_overlaps,
)


# Application configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
SLOTS_PATH = BASE_DIR / "parking_slots.json"
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
DETECTION_CONFIDENCE = 0.35
DIAGNOSTIC_THRESHOLDS = (0.05, 0.10, 0.15, 0.25)
YOLO_IMAGE_SIZE = 640
MAX_PROCESSING_WIDTH = 1280
SLOT_OVERLAP_THRESHOLD = 0.20
SMOOTHING_WINDOW = 5
SMOOTHING_OCCUPIED_VOTES = 3
ASPECT_RATIO_TOLERANCE = 0.02
DEBUG_MODE = True
DEBUG_LOG_INTERVAL_SECONDS = 1.0
CAMERA_FAILURE_LIMIT = 30
CAMERA_RETRY_DELAY = 0.03
PREVIEW_WINDOW = "SmartParking AI - Parking Detection"


class LatestFrameCamera:
    """Continuously drain the camera and keep only its newest frame."""

    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_frame = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="parking-camera-capture",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def _capture_loop(self):
        failures = 0
        while not self.stop_event.is_set():
            try:
                ok, frame = self.capture.read()
            except Exception as exc:
                with self.lock:
                    self.error = f"Kamera kadrini o'qishda xato: {exc}"
                return

            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= CAMERA_FAILURE_LIMIT:
                    with self.lock:
                        self.error = "Kamera oqimi uzildi yoki kadr kelmayapti."
                    return
                self.stop_event.wait(CAMERA_RETRY_DELAY)
                continue

            failures = 0
            with self.lock:
                self.latest_frame = frame.copy()
                self.sequence += 1

    def get_new_frame(self, last_sequence):
        with self.lock:
            if self.error:
                return None, self.sequence, self.error
            if self.latest_frame is None or self.sequence == last_sequence:
                return None, self.sequence, None
            return self.latest_frame.copy(), self.sequence, None

    def stop(self):
        self.stop_event.set()
        try:
            self.capture.release()
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


def resize_for_processing(frame):
    """Downscale while preserving aspect ratio; return a writable frame."""
    height, width = frame.shape[:2]
    if width <= MAX_PROCESSING_WIDTH:
        return frame.copy()
    scale = MAX_PROCESSING_WIDTH / width
    target = (MAX_PROCESSING_WIDTH, max(1, round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def extract_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], [], []
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    return coordinates, confidences, classes, boxes


def filter_detections(detections, class_ids):
    """Apply the vehicle class filter in Python so raw YOLO results stay visible."""
    boxes, confidences, classes, original_boxes = detections
    if len(classes) == 0:
        return [], [], [], original_boxes
    keep = np.isin(classes, np.asarray(class_ids, dtype=int))
    return boxes[keep], confidences[keep], classes[keep], original_boxes


def verify_model_weights(model):
    """Fail early if the local checkpoint is not the expected COCO detector."""
    expected = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    names = model.names
    missing = {
        class_id: expected_name
        for class_id, expected_name in expected.items()
        if names.get(class_id) != expected_name
    }
    class_count = getattr(model.model, "nc", None)
    if model.task != "detect" or class_count != 80 or missing:
        raise RuntimeError(
            f"YOLO weights COCO tekshiruvdan o'tmadi: task={model.task}, "
            f"nc={class_count}, noto'g'ri classlar={missing}"
        )
    print(f"YOLO weights verified: detect task, {class_count} COCO classes.")


def print_raw_detections(label, detections, names):
    boxes, confidences, classes, _ = detections
    print(f"\n{label}: {len(boxes)}")
    for box, confidence, class_id in zip(boxes, confidences, classes):
        class_name = names.get(int(class_id), str(class_id))
        coords = [round(float(value), 1) for value in box]
        print(
            f"Class: {class_name} | Confidence: {float(confidence):.2f} | "
            f"Bounding box: {coords}"
        )
    if not len(boxes):
        print("No detections at this confidence threshold.")


def vehicle_class_counts(detections, names):
    boxes, confidences, classes, _ = detections
    counts = {names[class_id]: 0 for class_id in VEHICLE_CLASS_IDS}
    for class_id in classes:
        name = names.get(int(class_id), str(class_id))
        counts[name] = counts.get(name, 0) + 1
    return len(boxes), counts


def detection_signature(detections):
    boxes, confidences, classes, _ = detections
    signatures = []
    for box, confidence, class_id in zip(boxes, confidences, classes):
        signatures.append((
            int(class_id),
            round(float(confidence), 4),
            tuple(round(float(value), 1) for value in box),
        ))
    return sorted(signatures)


def run_yolo_diagnostics(model, device):
    """Save a camera frame and independently sweep the saved image through YOLO."""
    capture = None
    try:
        capture = open_selected_camera()
        ok, camera_frame = capture.read()
        if not ok or camera_frame is None or camera_frame.size == 0:
            raise RuntimeError("iPhone kamerasidan diagnostika kadri olinmadi.")

        # Save the actual camera frame at full capture resolution. Inference below
        # reads this JPEG back, independently of the live camera pipeline.
        live_frame = resize_for_processing(camera_frame)
        if (live_frame.dtype != np.uint8 or live_frame.ndim != 3
                or live_frame.shape[2] != 3):
            raise RuntimeError(
                f"YOLO uchun noto'g'ri frame: shape={live_frame.shape}, "
                f"dtype={live_frame.dtype}"
            )

        saved_path = BASE_DIR / "parking_debug.jpg"
        if not cv2.imwrite(str(saved_path), camera_frame,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Diagnostika kadri saqlanmadi: {saved_path}")
        saved_frame = cv2.imread(str(saved_path), cv2.IMREAD_COLOR)
        if saved_frame is None:
            raise RuntimeError(f"Saqlangan kadr qayta o'qilmadi: {saved_path}")

        print(
            f"Captured camera frame: raw={camera_frame.shape[1]}x{camera_frame.shape[0]}, "
            f"live-YOLO input={live_frame.shape[1]}x{live_frame.shape[0]}, "
            f"saved JPEG={saved_frame.shape[1]}x{saved_frame.shape[0]}, "
            f"dtype={saved_frame.dtype}, channels=BGR"
        )
        print(f"Saved diagnostic frame: {saved_path}")
        print("Preprocess: full-resolution JPEG read by cv2.imread; YOLO handles letterboxing internally.")
        print("Model metadata:")
        print(f"  task={model.task}; names count={len(model.names)}; nc={model.model.nc}")
        print("  selected classes=" + ", ".join(
            f"{class_id}={model.names[class_id]}" for class_id in (2, 3, 5, 7, 67)
        ))
        print(f"  checkpoint={MODEL_PATH}; bytes={MODEL_PATH.stat().st_size}")
        print(f"  backend={device}")

        # The annotation uses the most permissive requested threshold and the
        # largest requested image size, so it includes every returned raw box.
        annotated_result = None
        for image_size in (640, 960, 1280):
            for confidence in DIAGNOSTIC_THRESHOLDS:
                result = model.predict(
                    source=saved_frame,
                    imgsz=image_size,
                    conf=confidence,
                    classes=None,
                    device=device,
                    verbose=False,
                )[0]
                detections = extract_detections(result)
                vehicles = filter_detections(detections, VEHICLE_CLASS_IDS)
                vehicle_count, counts = vehicle_class_counts(vehicles, model.names)
                print(
                    f"SAVED JPEG sweep imgsz={image_size} conf={confidence:.2f}: "
                    f"raw={len(detections[0])}; vehicles={vehicle_count} {counts}"
                )
                if image_size == 1280 and confidence == 0.05:
                    annotated_result = result
                    print_raw_detections(
                        "YOLO raw detections (saved JPEG, imgsz=1280, conf=0.05)",
                        detections,
                        model.names,
                    )

        annotated_path = BASE_DIR / "parking_debug_annotated.jpg"
        if annotated_result is None or not cv2.imwrite(
            str(annotated_path), annotated_result.plot(labels=True, conf=True, boxes=True),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"Annotatsiyalangan rasm saqlanmadi: {annotated_path}")
        print(f"Annotated raw detections image: {annotated_path}")
        return 0
    except Exception as exc:
        print(f"YOLO diagnostics failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if capture is not None:
            capture.release()


def log_raw_and_vehicle_debug(now, last_log, raw_detections,
                              vehicle_detections, names, overlaps,
                              slot_ratios, states, alignment_warning):
    if not DEBUG_MODE or now - last_log < DEBUG_LOG_INTERVAL_SECONDS:
        return last_log
    raw_boxes, raw_confidences, raw_classes, _ = raw_detections
    print(f"\nYOLO raw detections: {len(raw_boxes)}")
    for box, confidence, class_id in zip(raw_boxes, raw_confidences, raw_classes):
        class_name = names.get(int(class_id), str(class_id))
        coords = [round(float(value), 1) for value in box]
        print(
            f"Class: {class_name} | Confidence: {float(confidence):.2f} | "
            f"Bounding box: {coords}"
        )

    vehicle_boxes, vehicle_confidences, vehicle_classes, _ = vehicle_detections
    counts = {name: 0 for name in ("car", "motorcycle", "bus", "truck")}
    for class_id in vehicle_classes:
        class_name = names.get(int(class_id), str(class_id))
        counts[class_name] = counts.get(class_name, 0) + 1
    print(f"Detected vehicles: {len(vehicle_boxes)} {counts}")
    occupied = sum(states.values())
    available = len(states) - occupied
    print(
        f"Occupied parking slots: {occupied}; "
        f"Available parking slots: {available}"
    )
    visible_overlaps = [
        f"vehicle {item['vehicle_index'] + 1}->{item['slot_id']} "
        f"{item['ratio']:.1%}"
        for item in overlaps if item["ratio"] >= 0.01
    ]
    print(
        f"Vehicle-slot intersections: {visible_overlaps or 'none'}; "
        f"slot ratios={slot_ratios}; "
        f"alignment={'CHECK' if alignment_warning else 'OK'}"
    )
    return now


def annotate_frame(frame, slot_masks, states, slot_ratios,
                   detections, names, fps, alignment_warning=False):
    # Tint each parking polygon before drawing its bright outline.
    overlay = frame.copy()
    for slot in slot_masks:
        ratio = slot_ratios[slot.slot_id]
        raw_state = ratio >= SLOT_OVERLAP_THRESHOLD
        uncertain = (
            alignment_warning
            or raw_state != states[slot.slot_id]
            or (not states[slot.slot_id]
                and ratio >= SLOT_OVERLAP_THRESHOLD * 0.75)
        )
        color = ((0, 255, 255) if uncertain else
                 ((0, 0, 255) if states[slot.slot_id] else (0, 200, 0)))
        cv2.fillPoly(overlay, [slot.points], color)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    for slot in slot_masks:
        occupied = states[slot.slot_id]
        ratio = slot_ratios[slot.slot_id]
        raw_state = ratio >= SLOT_OVERLAP_THRESHOLD
        uncertain = (
            alignment_warning
            or raw_state != occupied
            or (not occupied and ratio >= SLOT_OVERLAP_THRESHOLD * 0.75)
        )
        color = ((0, 255, 255) if uncertain else
                 ((0, 0, 255) if occupied else (0, 255, 0)))
        status = "UNCERTAIN" if uncertain else ("OCCUPIED" if occupied else "AVAILABLE")
        center = np.mean(slot.points, axis=0).astype(int)
        cv2.polylines(frame, [slot.points], True, color, 3, cv2.LINE_AA)
        cv2.putText(
            frame, f"{slot.slot_id} {status} {ratio:.0%}", tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA,
        )

    boxes, confidences, classes, _ = detections
    for box, confidence, class_id in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        class_name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
        label = f"{class_name} {float(confidence):.2f}"
        # Blue in BGR for each vehicle box; confidence and class stay visible.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame, label, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA,
        )

    total = len(slot_masks)
    occupied_count = sum(states.values())
    available_count = total - occupied_count
    occupancy_percent = occupied_count * 100.0 / total if total else 0.0
    uncertain_count = sum(
        alignment_warning
        or (slot_ratios[slot_id] >= SLOT_OVERLAP_THRESHOLD) != state
        or (not state and slot_ratios[slot_id] >= SLOT_OVERLAP_THRESHOLD * 0.75)
        for slot_id, state in states.items()
    )
    cv2.rectangle(frame, (10, 10), (630, 190), (25, 25, 25), -1)
    rows = [
        (f"Vehicles: {len(boxes)}", (255, 255, 255)),
        (f"Total: {total}  Occupied: {occupied_count}  Available: {available_count}", (255, 255, 255)),
        (f"Uncertain: {uncertain_count}  Occupancy: {occupancy_percent:.1f}%", (0, 220, 255)),
        (f"FPS: {fps:.1f}", (255, 255, 0)),
    ]
    for row, (text, color) in enumerate(rows):
        cv2.putText(
            frame, text, (20, 38 + row * 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
        )
    return frame


def run(diagnose_yolo=False):
    stream = None
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO modeli topilmadi: {MODEL_PATH}")
        source_width, source_height, slots = load_slots(SLOTS_PATH)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"YOLO11n yuklanmoqda: {MODEL_PATH.name} ({device})")
        model = YOLO(str(MODEL_PATH))
        verify_model_weights(model)
        if diagnose_yolo:
            return run_yolo_diagnostics(model, device)
        try:
            camera = open_selected_camera()
        except RuntimeError as exc:
            raise RuntimeError(f"iPhone kamerasi ochilmadi: {exc}") from exc

        stream = LatestFrameCamera(camera)
        stream.start()
        print("Parking occupancy ishga tushdi. Chiqish uchun Q bosing.")
        print(f"Slotlar: {len(slots)} | overlap threshold: {SLOT_OVERLAP_THRESHOLD:.0%} | "
              f"smoothing: {SMOOTHING_OCCUPIED_VOTES}/{SMOOTHING_WINDOW} kadr")
        print("COCO vehicle IDs: " + ", ".join(
            f"{class_id}={model.names[class_id]}" for class_id in VEHICLE_CLASS_IDS
        ))

        last_sequence = 0
        slot_masks = None
        smoother = None
        previous_time = time.perf_counter()
        last_debug_log = 0.0
        fps = 0.0
        active_device = device
        alignment_warning = False
        checked_frame_size = None

        while True:
            frame, sequence, stream_error = stream.get_new_frame(last_sequence)
            if stream_error:
                print(stream_error)
                break
            if frame is None:
                time.sleep(0.005)
                continue
            last_sequence = sequence
            frame = resize_for_processing(frame)
            frame_height, frame_width = frame.shape[:2]

            if checked_frame_size != (frame_width, frame_height):
                saved_aspect = source_width / source_height
                camera_aspect = frame_width / frame_height
                aspect_delta = abs(camera_aspect / saved_aspect - 1.0)
                alignment_warning = aspect_delta > ASPECT_RATIO_TOLERANCE
                checked_frame_size = (frame_width, frame_height)
                print(
                    f"Slot coordinate source={source_width}x{source_height}; "
                    f"camera processing frame={frame_width}x{frame_height}; "
                    f"aspect difference={aspect_delta:.1%}."
                )
                if alignment_warning:
                    print(
                        "WARNING: camera/frame aspect ratio differs from the saved "
                        "slot view. Polygons are shown yellow; verify orientation "
                        "and re-edit slots from this camera if needed. JSON was not changed."
                    )

            if slot_masks is None or slot_masks[0].mask.shape != (frame_height, frame_width):
                slot_masks = build_slot_masks(
                    slots,
                    (source_width, source_height),
                    (frame_width, frame_height),
                )
                smoother = TemporalOccupancy(
                    [slot.slot_id for slot in slot_masks],
                    window_size=SMOOTHING_WINDOW,
                    occupied_votes=SMOOTHING_OCCUPIED_VOTES,
                )

            try:
                result = model.predict(
                    source=frame,
                    imgsz=YOLO_IMAGE_SIZE,
                    conf=DETECTION_CONFIDENCE,
                    classes=None,
                    device=active_device,
                    verbose=False,
                )[0]
            except Exception as exc:
                if active_device != "mps":
                    raise
                print(f"MPS inference xatosi ({exc}); CPU'ga o'tilmoqda.")
                active_device = "cpu"
                result = model.predict(
                    source=frame,
                    imgsz=YOLO_IMAGE_SIZE,
                    conf=DETECTION_CONFIDENCE,
                    classes=None,
                    device=active_device,
                    verbose=False,
                )[0]

            raw_detections = extract_detections(result)
            detections = filter_detections(raw_detections, VEHICLE_CLASS_IDS)
            vehicle_slot_overlaps = slot_vehicle_overlaps(slot_masks, detections[0])
            ratios = {slot.slot_id: 0.0 for slot in slot_masks}
            for overlap in vehicle_slot_overlaps:
                slot_id = overlap["slot_id"]
                ratios[slot_id] = max(ratios[slot_id], overlap["ratio"])
            states = smoother.update(ratios, SLOT_OVERLAP_THRESHOLD)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps
            previous_time = now

            display = annotate_frame(
                frame, slot_masks, states, ratios, detections, model.names, fps,
                alignment_warning=alignment_warning,
            )
            last_debug_log = log_raw_and_vehicle_debug(
                now, last_debug_log, raw_detections, detections,
                model.names, vehicle_slot_overlaps, ratios, states,
                alignment_warning,
            )
            cv2.imshow(PREVIEW_WINDOW, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        return 0
    except KeyboardInterrupt:
        print("\nDastur foydalanuvchi tomonidan to'xtatildi.")
        return 0
    except Exception as exc:
        print(f"Parking detection xatosi: {exc}")
        return 1
    finally:
        if stream is not None:
            stream.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnose-yolo", action="store_true",
        help="Capture one selected-camera frame and compare raw YOLO outputs.",
    )
    arguments = parser.parse_args()
    raise SystemExit(run(diagnose_yolo=arguments.diagnose_yolo))
