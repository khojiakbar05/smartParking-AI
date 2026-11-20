"""Offline comparison of box-overlap and bottom-center parking assignments.

Runs the existing YOLO checkpoint on saved frame_NN.jpg images, then writes
side-by-side human-review artifacts without changing production configuration.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from parking_logic import build_slot_masks, load_slots, slot_vehicle_overlaps


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "debug" / "camera_20261119_112904"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "debug"
MODEL_PATH = BASE_DIR / "yolo11n.pt"
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
CONFIDENCE = 0.35
IMAGE_SIZE = 640
OVERLAP_THRESHOLD = 0.20


def unique_output_dir(root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"assignment_comparison_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"assignment_comparison_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def detection_records(result, names):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    return [
        {"box": [float(v) for v in box], "confidence": float(score),
         "class_id": int(class_id), "class_name": str(names[int(class_id)])}
        for box, score, class_id in zip(xyxy, conf, classes)
        if int(class_id) in VEHICLE_CLASS_IDS
    ]


def method_b_assignments(slot_masks, vehicles):
    assignments = {index: [] for index in range(len(vehicles))}
    for index, vehicle in enumerate(vehicles):
        x1, y1, x2, y2 = vehicle["box"]
        point = ((x1 + x2) / 2.0, y2)
        for slot in slot_masks:
            if cv2.pointPolygonTest(slot.points, point, False) >= 0:
                assignments[index].append(slot.slot_id)
    return assignments


def draw_overlay(frame, slot_masks, vehicles, assignments, method, ratios=None):
    canvas = frame.copy()
    assigned_slots = {slot for slots in assignments.values() for slot in slots}
    for slot in slot_masks:
        color = (30, 30, 230) if slot.slot_id in assigned_slots else (40, 210, 40)
        cv2.polylines(canvas, [slot.points], True, color, 2, cv2.LINE_AA)
        center = np.mean(slot.points, axis=0).astype(int)
        suffix = f" {ratios[slot.slot_id]:.0%}" if ratios is not None else ""
        cv2.putText(canvas, slot.slot_id + suffix, tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    for index, vehicle in enumerate(vehicles):
        x1, y1, x2, y2 = np.rint(vehicle["box"]).astype(int)
        color = (255, 130, 20)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        point = (int(round((vehicle["box"][0] + vehicle["box"][2]) / 2)),
                 int(round(vehicle["box"][3])))
        cv2.circle(canvas, point, 6, (255, 0, 255), -1, cv2.LINE_AA)
        slots = assignments[index]
        label = f"V{index + 1}: {','.join(slots) if slots else 'none'}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, method, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def write_p22_inspection(frame, slot_masks, vehicles, a_assignments,
                         b_assignments, output_dir, stem):
    p22 = next(slot for slot in slot_masks if slot.slot_id == "P22")
    points = p22.points
    pad = 100
    x0 = max(0, int(points[:, 0].min()) - pad)
    y0 = max(0, int(points[:, 1].min()) - pad)
    x1 = min(frame.shape[1], int(points[:, 0].max()) + pad)
    y1 = min(frame.shape[0], int(points[:, 1].max()) + pad)
    # Crop the overlays and emphasize P22 with a yellow outline for focused review.
    for method, assignments in (("A", a_assignments), ("B", b_assignments)):
        ratios = None
        overlay = draw_overlay(frame, slot_masks, vehicles, assignments,
                               f"Method {method} — P22 area", ratios)
        cv2.polylines(overlay, [p22.points], True, (0, 255, 255), 5, cv2.LINE_AA)
        center = np.mean(p22.points, axis=0).astype(int)
        cv2.putText(overlay, "P22 INSPECT", tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 3, cv2.LINE_AA)
        crop = overlay[y0:y1, x0:x1]
        path = output_dir / f"{stem}_method_{method}_P22_crop.jpg"
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--slots", type=Path, default=BASE_DIR / "parking_slots.json")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    frame_paths = sorted(args.input_dir.glob("frame_[0-9][0-9].jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frame_NN.jpg files found in {args.input_dir}")
    if not args.model.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {args.model}")

    from ultralytics import YOLO

    source_width, source_height, slot_defs = load_slots(args.slots)
    if len(slot_defs) != 22:
        raise ValueError(f"Expected active 22-slot configuration; found {len(slot_defs)}")
    if [slot_id for slot_id, _ in slot_defs] != [f"P{i:02d}" for i in range(1, 23)]:
        raise ValueError("Expected slot IDs P01 through P22 in order")
    model = YOLO(str(args.model))
    result_dir = unique_output_dir(args.output_root)
    all_frames = []

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read {frame_path}")
        h, w = frame.shape[:2]
        masks = build_slot_masks(slot_defs, (source_width, source_height), (w, h))
        result = model.predict(source=frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                               classes=None, verbose=False)[0]
        vehicles = detection_records(result, model.names)
        boxes = [v["box"] for v in vehicles]

        overlaps = slot_vehicle_overlaps(masks, boxes)
        ratios = {slot.slot_id: 0.0 for slot in masks}
        a_assignments = {index: [] for index in range(len(vehicles))}
        a_ratios = {index: {} for index in range(len(vehicles))}
        for overlap in overlaps:
            slot_id = overlap["slot_id"]
            vehicle_index = overlap["vehicle_index"]
            ratios[slot_id] = max(ratios[slot_id], overlap["ratio"])
            a_ratios[vehicle_index][slot_id] = overlap["ratio"]
            if overlap["ratio"] >= OVERLAP_THRESHOLD:
                a_assignments[vehicle_index].append(slot_id)
        b_assignments = method_b_assignments(masks, vehicles)

        stem = frame_path.stem
        overlays = {
            "A": draw_overlay(frame, masks, vehicles, a_assignments,
                              "Method A — box overlap (20% slot area)", ratios),
            "B": draw_overlay(frame, masks, vehicles, b_assignments,
                              "Method B — bottom-center point", None),
        }
        for method, image in overlays.items():
            out_path = result_dir / f"{stem}_method_{method}.jpg"
            if not cv2.imwrite(str(out_path), image):
                raise RuntimeError(f"Could not save {out_path}")
        write_p22_inspection(frame, masks, vehicles, a_assignments,
                             b_assignments, result_dir, stem)

        vehicle_rows = []
        for index, vehicle in enumerate(vehicles):
            a_slots = sorted(a_assignments[index])
            b_slots = sorted(b_assignments[index])
            vehicle_rows.append({
                "vehicle_id": f"V{index + 1}",
                **vehicle,
                "bottom_center": [(vehicle["box"][0] + vehicle["box"][2]) / 2.0,
                                  vehicle["box"][3]],
                "method_a_slots": a_slots,
                "method_a_ratios": a_ratios[index],
                "method_a_multiple_slots": len(a_slots) > 1,
                "method_b_slots": b_slots,
                "method_b_unassigned": len(b_slots) == 0,
            })
        all_frames.append({
            "frame": str(frame_path.resolve()),
            "width": w,
            "height": h,
            "vehicles": vehicle_rows,
            "method_a_unoccupied_by_threshold": [
                sid for sid, ratio in ratios.items() if ratio < OVERLAP_THRESHOLD
            ],
            "slot_overlap_ratios_max": ratios,
        })

    report = {
        "active_slots_file": str(args.slots.resolve()),
        "slot_count": len(slot_defs),
        "slot_ids": [sid for sid, _ in slot_defs],
        "slot_source_resolution": [source_width, source_height],
        "model": str(args.model.resolve()),
        "confidence_threshold": CONFIDENCE,
        "image_size": IMAGE_SIZE,
        "method_a_occupied_threshold": OVERLAP_THRESHOLD,
        "method_a": "bbox/rasterized polygon intersection divided by slot area; assigned at >=20%",
        "method_b": "bottom-center point (center x, y2) inside polygon; boundary included",
        "smoothing": {"window": 5, "occupied_votes": 3,
                      "applied": False, "reason": "per-frame spatial assignment comparison"},
        "ground_truth": "No manual labels supplied; this report does not establish accuracy.",
        "vehicle_detection_note": (
            "No vehicle-class detections were returned on these frames at the "
            "configured confidence threshold; assignment lists are consequently empty."
            if sum(len(frame["vehicles"]) for frame in all_frames) == 0 else None
        ),
        "frames": all_frames,
    }
    report_path = result_dir / "assignment_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = result_dir / "vehicle_assignments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["frame", "vehicle_id", "class_name", "confidence", "box",
                  "bottom_center", "method_a_slots", "method_a_ratios",
                  "method_a_multiple_slots", "method_b_slots", "method_b_unassigned"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_report in all_frames:
            for vehicle in frame_report["vehicles"]:
                writer.writerow({"frame": Path(frame_report["frame"]).name,
                                 **{key: json.dumps(vehicle[key]) if isinstance(vehicle[key], (list, dict))
                                    else vehicle[key] for key in fields if key != "frame"}})
    print(f"Output directory: {result_dir.resolve()}")
    print(f"JSON summary: {report_path.resolve()}")
    print(f"CSV summary: {csv_path.resolve()}")
    print(f"Frames processed: {len(all_frames)}; slots: {len(slot_defs)}")


if __name__ == "__main__":
    main()
