"""Parking slot geometry and temporal occupancy smoothing."""

import json
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
        if slot_id in seen_ids:
            raise ValueError(f"Slot ID takrorlangan: {slot_id}")
        seen_ids.add(slot_id)
        validated.append((slot_id, points))
    return source_width, source_height, validated


def build_slot_masks(slots, source_size, frame_size):
    """Scale saved coordinates independently in X/Y and rasterize polygons."""
    source_width, source_height = source_size
    frame_width, frame_height = frame_size
    if min(source_width, source_height, frame_width, frame_height) <= 0:
        raise ValueError("Kadr va slot o'lchamlari musbat bo'lishi kerak.")

    scale = np.array([frame_width / source_width,
                      frame_height / source_height], dtype=np.float32)
    slot_masks = []
    for slot_id, source_points in slots:
        points = np.rint(source_points * scale).astype(np.int32)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        area = int(cv2.countNonZero(mask))
        if area == 0:
            raise ValueError(f"{slot_id}: slot poligoni joriy kadrda bo'sh.")
        slot_masks.append(SlotMask(slot_id, points, mask, area))
    return slot_masks


def slot_overlap_ratios(slot_masks, vehicle_boxes):
    """Return maximum vehicle-box intersection / slot-area for each slot."""
    if not slot_masks:
        return {}
    frame_height, frame_width = slot_masks[0].mask.shape
    ratios = {slot.slot_id: 0.0 for slot in slot_masks}

    for x1, y1, x2, y2 in vehicle_boxes:
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
            ratio = overlap_pixels / slot.area
            if ratio > ratios[slot.slot_id]:
                ratios[slot.slot_id] = ratio
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
