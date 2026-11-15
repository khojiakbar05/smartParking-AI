"""Manually label actual-camera slot crops for OCCUPIED/VACANT training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np

from camera_devices import open_selected_camera
from parking_logic import build_slot_masks, load_slots
from slot_crops import crop_slots


LABEL_KEYS = {ord("o"): "occupied", ord("v"): "vacant"}


def _crop_hash(image):
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _existing_samples(dataset_dir):
    known = {}
    for label in ("occupied", "vacant"):
        folder = dataset_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.glob("*.jpg")):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable existing sample: {path}")
                continue
            known[_crop_hash(image)] = (label, path)
    return known


def _save_sample(crop, label, dataset_dir, slot_id, frame_size, known):
    encoded_ok, encoded = cv2.imencode(
        ".jpg", crop.image, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not encoded_ok:
        raise OSError("JPEG encoding failed; sample was not saved.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0 or decoded.ndim != 3:
        raise OSError("Encoded sample did not pass image validation.")
    sample_hash = _crop_hash(decoded)
    if sample_hash in known:
        old_label, old_path = known[sample_hash]
        if old_label != label:
            print(
                f"Duplicate crop has conflicting label ({old_label} at {old_path}); "
                "sample not saved."
            )
        else:
            print(f"Exact duplicate already saved at {old_path}; sample not saved.")
        return None

    target_dir = dataset_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", slot_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = target_dir / f"{safe_id}_{stamp}_{sample_hash[:10]}.jpg"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target_dir, suffix=".jpg", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded.tobytes())
            temporary.flush()
            os.fsync(temporary.fileno())
        check = cv2.imread(str(temporary_path), cv2.IMREAD_COLOR)
        if check is None or check.shape != decoded.shape:
            raise OSError("Temporary sample failed decode/shape validation.")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    manifest_entry = {
        "path": str(target.relative_to(dataset_dir)),
        "slot_id": slot_id,
        "label": label.upper(),
        "sha256_crop": sample_hash,
        "frame_width": int(frame_size[0]),
        "frame_height": int(frame_size[1]),
        "crop_method": crop.method,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_source": "manual_keyboard",
    }
    manifest = dataset_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    known[sample_hash] = (label, target)
    print(f"Saved manually labeled {label.upper()} sample: {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=Path, default=Path(__file__).with_name("parking_slots.json"))
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path(__file__).with_name("dataset") / "slot_samples")
    parser.add_argument("--no-perspective", action="store_true",
                        help="Use polygon-masked bounding crops instead of quadrilateral warps.")
    args = parser.parse_args()
    source_width, source_height, slot_defs = load_slots(args.slots)
    known = _existing_samples(args.dataset_dir)
    camera = None
    try:
        camera = open_selected_camera()
        print(f"Selected camera opened; slots={len(slot_defs)}.")
        print("Keyboard: O=label selected crop OCCUPIED; V=VACANT; N/P=next/previous slot; Q=quit.")
        print("Only the user's explicit keypress saves a label. YOLO is not used.")
        selected = 0
        masks = None
        crops = []
        frame_size = None
        window_name = "SmartParking - Manual Slot Sample Collector"
        crop_window = "Selected slot crop (manual review)"

        while True:
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError("Selected camera stopped returning valid frames.")
            height, width = frame.shape[:2]
            if frame_size != (width, height):
                masks = build_slot_masks(
                    slot_defs, (source_width, source_height), (width, height)
                )
                frame_size = (width, height)
            crops = crop_slots(frame, masks, perspective=not args.no_perspective)
            selected %= len(crops)

            display = frame.copy()
            for index, mask in enumerate(masks):
                color = (0, 255, 255) if index == selected else (255, 180, 0)
                cv2.polylines(display, [mask.points], True, color,
                              4 if index == selected else 2, cv2.LINE_AA)
                center = np.mean(mask.points, axis=0).astype(int)
                cv2.putText(display, mask.slot_id, tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            current = crops[selected]
            header = f"Selected {current.slot_id}  |  O=OCCUPIED  V=VACANT  N/P=change  Q=quit"
            cv2.rectangle(display, (8, 8), (min(width - 8, 1040), 44), (25, 25, 25), -1)
            cv2.putText(display, header, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            cv2.imshow(crop_window, current.image)

            key = cv2.waitKey(1) & 0xFF
            if key in LABEL_KEYS:
                _save_sample(
                    current, LABEL_KEYS[key], args.dataset_dir,
                    current.slot_id, frame_size, known,
                )
            elif key in (ord("n"), ord("."), 83):
                selected = (selected + 1) % len(crops)
            elif key in (ord("p"), ord(","), 81):
                selected = (selected - 1) % len(crops)
            elif key in (ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        print("\nCollection interrupted by the user.")
    finally:
        if camera is not None:
            camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
