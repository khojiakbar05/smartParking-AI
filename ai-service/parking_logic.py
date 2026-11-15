"""Parking slot geometry and temporal occupancy smoothing."""

import json
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SlotMask:
    slot_id: str
    points: np.ndarray
    mask: np.ndarray
    area: int


def load_slots(path):
    """Read and validate the saved slot editor JSON format."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Parking slot faylini o'qib bo'lmadi: {exc}") from exc

    try:
        source_width = int(data["frame_width"])
        source_height = int(data["frame_height"])
        slots = data["slots"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Slot JSON'da frame_width, frame_height va slots bo'lishi kerak.") from exc

    if source_width <= 0 or source_height <= 0 or not isinstance(slots, list) or not slots:
        raise ValueError("Slot o'lchamlari musbat bo'lishi va kamida bitta slot bo'lishi kerak.")

    validated = []
    seen_ids = set()
    for number, slot in enumerate(slots, start=1):
        if not isinstance(slot, dict):
            raise ValueError(f"{number}-slot noto'g'ri formatda.")
        slot_id = str(slot.get("id", f"P{number:02d}"))
        try:
            points = np.asarray(slot["points"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{slot_id}: nuqtalarni o'qib bo'lmadi.") from exc
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"{slot_id}: kamida 3 ta (x, y) nuqta kerak.")
        if not np.isfinite(points).all():
            raise ValueError(f"{slot_id}: koordinatalar son bo'lishi kerak.")
        if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= source_width) or
                np.any(points[:, 1] < 0) or np.any(points[:, 1] >= source_height)):
            raise ValueError(
                f"{slot_id}: nuqta parking_slots.json dagi "
                f"{source_width}x{source_height} kadr chegarasidan tashqarida."
            )
        if slot_id in seen_ids:
            raise ValueError(f"Slot ID takrorlangan: {slot_id}")
        seen_ids.add(slot_id)

        if _has_crossing_edges(points):
            # A four-corner parking bay should be a simple contour. P03 in the
            # existing file has its last two corners reversed (a bow-tie).
            center = points.mean(axis=0)
            angles = np.arctan2(points[:, 1] - center[1],
                                points[:, 0] - center[0])
            candidate = points[np.argsort(angles)]
            if not cv2.isContourConvex(candidate.astype(np.int32)):
                raise ValueError(
                    f"{slot_id}: poligon kesishadi va xavfsiz tartiblanmadi; "
                    "slot editorida nuqtalarni qayta belgilang."
                )
            warnings.warn(
                f"{slot_id}: o'zaro kesishgan burchaklar aniqlanib, faqat "
                "joriy ishga tushirish uchun tartiblandi. JSON fayl o'zgartirilmadi.",
                RuntimeWarning,
            )
            points = candidate
        validated.append((slot_id, points))
    return source_width, source_height, validated


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _has_crossing_edges(points):
    """Detect proper intersections between non-adjacent polygon edges."""
    count = len(points)
    epsilon = 1e-6
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or second == (first + 1) % count:
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
            cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
            if (ab_c * ab_d < -epsilon and cd_a * cd_b < -epsilon):
                return True
    return False


def build_slot_masks(slots, source_size, frame_size):
    """Aspect-preserving center-fit of saved coordinates, then rasterize."""
    source_width, source_height = source_size
    frame_width, frame_height = frame_size
    if min(source_width, source_height, frame_width, frame_height) <= 0:
        raise ValueError("Kadr va slot o'lchamlari musbat bo'lishi kerak.")

    scale = min(frame_width / source_width, frame_height / source_height)
    offset = np.array([
        (frame_width - source_width * scale) / 2,
        (frame_height - source_height * scale) / 2,
    ], dtype=np.float32)
    slot_masks = []
    for slot_id, source_points in slots:
        points = np.rint(source_points * scale + offset).astype(np.int32)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        area = int(cv2.countNonZero(mask))
        if area == 0:
            raise ValueError(f"{slot_id}: slot poligoni joriy kadrda bo'sh.")
        slot_masks.append(SlotMask(slot_id, points, mask, area))
    return slot_masks


def slot_vehicle_overlaps(slot_masks, vehicle_boxes):
    """Return every positive vehicle/polygon intersection and its slot ratio."""
    if not slot_masks:
        return []
    frame_height, frame_width = slot_masks[0].mask.shape
    overlaps = []

    for vehicle_index, (x1, y1, x2, y2) in enumerate(vehicle_boxes):
        left = max(0, min(frame_width, int(np.floor(x1))))
        top = max(0, min(frame_height, int(np.floor(y1))))
        right = max(0, min(frame_width, int(np.ceil(x2))))
        bottom = max(0, min(frame_height, int(np.ceil(y2))))
        if right <= left or bottom <= top:
            continue
        for slot in slot_masks:
            overlap_pixels = cv2.countNonZero(
                slot.mask[top:bottom, left:right]
            )
            if overlap_pixels:
                overlaps.append({
                    "vehicle_index": vehicle_index,
                    "slot_id": slot.slot_id,
                    "intersection_area": overlap_pixels,
                    "slot_area": slot.area,
                    "ratio": overlap_pixels / slot.area,
                })
    return overlaps


def slot_overlap_ratios(slot_masks, vehicle_boxes):
    """Return maximum vehicle-box intersection / slot-area for each slot."""
    ratios = {slot.slot_id: 0.0 for slot in slot_masks}
    for overlap in slot_vehicle_overlaps(slot_masks, vehicle_boxes):
        slot_id = overlap["slot_id"]
        ratios[slot_id] = max(ratios[slot_id], overlap["ratio"])
    return ratios


class TemporalOccupancy:
    """Majority vote over recent per-slot occupancy observations."""

    def __init__(self, slot_ids, window_size=5, occupied_votes=3):
        if window_size < 1 or not 1 <= occupied_votes <= window_size:
            raise ValueError("Smoothing window/votes sozlamasi noto'g'ri.")
        self.occupied_votes = occupied_votes
        self.history = {
            slot_id: deque(maxlen=window_size) for slot_id in slot_ids
        }

    def update(self, ratios, overlap_threshold):
        states = {}
        for slot_id, history in self.history.items():
            history.append(ratios.get(slot_id, 0.0) >= overlap_threshold)
            states[slot_id] = sum(history) >= self.occupied_votes
        return states
