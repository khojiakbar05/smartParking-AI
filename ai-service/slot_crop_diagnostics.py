"""Build human-reviewable parking-slot crops from saved camera frames."""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from parking_logic import build_slot_masks, load_slots
from slot_crops import crop_slots, save_debug_crops


def _contact_sheet(crops, path, cell_size=260, columns=4):
    rows = max(1, math.ceil(len(crops) / columns))
    sheet = np.full((rows * cell_size, columns * cell_size, 3), 32, dtype=np.uint8)
    for index, crop in enumerate(crops):
        row, column = divmod(index, columns)
        image = crop.image
        scale = min((cell_size - 16) / max(image.shape[1], 1),
                    (cell_size - 44) / max(image.shape[0], 1))
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)),
             max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        x = column * cell_size + (cell_size - resized.shape[1]) // 2
        y = row * cell_size + 28 + (cell_size - 36 - resized.shape[0]) // 2
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(
            sheet, f"{crop.slot_id} ({crop.method})",
            (column * cell_size + 8, row * cell_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not save crop contact sheet: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--slots", type=Path, default=Path(__file__).with_name("parking_slots.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("debug") / "slot_crops")
    args = parser.parse_args()

    source_width, source_height, slot_defs = load_slots(args.slots)
    frames = sorted(args.input_dir.glob("frame_[0-9][0-9].jpg"))
    if not frames:
        raise RuntimeError(f"No frame_NN.jpg images found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Slots: {len(slot_defs)} from {source_width}x{source_height} coordinates")

    for path in frames:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read frame: {path}")
        masks = build_slot_masks(
            slot_defs, (source_width, source_height), (frame.shape[1], frame.shape[0])
        )
        crops = crop_slots(frame, masks, perspective=True)
        sample_dir = args.output_dir / path.stem
        save_debug_crops(crops, sample_dir, prefix=path.stem)
        _contact_sheet(crops, args.output_dir / f"{path.stem}_contact.jpg")
        overlay = frame.copy()
        for slot in masks:
            cv2.polylines(overlay, [slot.points], True, (0, 255, 255), 2, cv2.LINE_AA)
            center = np.mean(slot.points, axis=0).astype(int)
            cv2.putText(overlay, slot.slot_id, tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA)
        overlay_path = args.output_dir / f"{path.stem}_polygons.jpg"
        if not cv2.imwrite(str(overlay_path), overlay,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not save polygon overlay: {overlay_path}")
        print(f"{path.name}: {len(crops)} crops saved to {sample_dir}")
        for crop in crops:
            print(f"  {crop.slot_id}: {crop.image.shape[1]}x{crop.image.shape[0]} ({crop.method})")
        print(f"  contact sheet: {args.output_dir / (path.stem + '_contact.jpg')}")
        print(f"  polygon alignment view: {overlay_path}")


if __name__ == "__main__":
    main()
