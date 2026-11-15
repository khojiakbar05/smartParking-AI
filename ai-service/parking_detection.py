"""Real-time YOLO11n parking occupancy detection."""

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
    slot_overlap_ratios,
)


# Application configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
SLOTS_PATH = BASE_DIR / "parking_slots.json"
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
DETECTION_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 640
MAX_PROCESSING_WIDTH = 1280
SLOT_OVERLAP_THRESHOLD = 0.20
SMOOTHING_WINDOW = 5
SMOOTHING_OCCUPIED_VOTES = 3
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


def extract_vehicle_boxes(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], [], [], []
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    return coordinates, confidences, classes, boxes


def annotate_frame(frame, slot_masks, states, slot_ratios,
                   detections, names, fps):
    # Tint each parking polygon before drawing its bright outline.
    overlay = frame.copy()
    for slot in slot_masks:
        color = (0, 0, 255) if states[slot.slot_id] else (0, 200, 0)
        cv2.fillPoly(overlay, [slot.points], color)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    for slot in slot_masks:
        occupied = states[slot.slot_id]
        color = (0, 0, 255) if occupied else (0, 255, 0)
        status = "OCCUPIED" if occupied else "AVAILABLE"
        center = np.mean(slot.points, axis=0).astype(int)
        cv2.polylines(frame, [slot.points], True, color, 3, cv2.LINE_AA)
        cv2.putText(
            frame, f"{slot.slot_id} {status}", tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    boxes, confidences, classes, _ = detections
    for box, confidence, class_id in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        class_name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
        label = f"{class_name} {float(confidence):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 190, 0), 2)
        cv2.putText(
            frame, label, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 0), 2, cv2.LINE_AA,
        )

    total = len(slot_masks)
    occupied_count = sum(states.values())
    available_count = total - occupied_count
    occupancy_percent = occupied_count * 100.0 / total if total else 0.0
    cv2.rectangle(frame, (10, 10), (455, 150), (25, 25, 25), -1)
    rows = [
        (f"Total: {total}   Occupied: {occupied_count}   Available: {available_count}",
         (255, 255, 255)),
        (f"Occupancy: {occupancy_percent:.1f}%", (0, 220, 255)),
        (f"FPS: {fps:.1f}", (255, 255, 0)),
    ]
    for row, (text, color) in enumerate(rows):
        cv2.putText(
            frame, text, (20, 45 + row * 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA,
        )
    return frame


def run():
    stream = None
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO modeli topilmadi: {MODEL_PATH}")
        source_width, source_height, slots = load_slots(SLOTS_PATH)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"YOLO11n yuklanmoqda: {MODEL_PATH.name} ({device})")
        model = YOLO(str(MODEL_PATH))
        try:
            camera = open_selected_camera()
        except RuntimeError as exc:
            raise RuntimeError(f"iPhone kamerasi ochilmadi: {exc}") from exc

        stream = LatestFrameCamera(camera)
        stream.start()
        print("Parking occupancy ishga tushdi. Chiqish uchun Q bosing.")
        print(f"Slotlar: {len(slots)} | overlap threshold: {SLOT_OVERLAP_THRESHOLD:.0%} | "
              f"smoothing: {SMOOTHING_OCCUPIED_VOTES}/{SMOOTHING_WINDOW} kadr")

        last_sequence = 0
        slot_masks = None
        smoother = None
        previous_time = time.perf_counter()
        fps = 0.0
        active_device = device

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
                    classes=VEHICLE_CLASS_IDS,
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
                    classes=VEHICLE_CLASS_IDS,
                    device=active_device,
                    verbose=False,
                )[0]

            detections = extract_vehicle_boxes(result)
            ratios = slot_overlap_ratios(slot_masks, detections[0])
            states = smoother.update(ratios, SLOT_OVERLAP_THRESHOLD)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps
            previous_time = now

            display = annotate_frame(
                frame, slot_masks, states, ratios, detections, model.names, fps
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
    raise SystemExit(run())
