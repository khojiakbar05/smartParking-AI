"""Mask and perspective-aware crops for saved parking-slot polygons."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SlotCrop:
    slot_id: str
    image: np.ndarray
    mask: np.ndarray
    method: str


def _ordered_quad(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not cv2.isContourConvex(points.astype(np.int32)):
        return None
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    # Ensure top-left, top-right, bottom-right, bottom-left order.
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def _perspective_crop(frame, slot_id, points):
    quad = _ordered_quad(points)
    if quad is None:
        return None
    top_width = np.linalg.norm(quad[1] - quad[0])
    bottom_width = np.linalg.norm(quad[2] - quad[3])
    left_height = np.linalg.norm(quad[3] - quad[0])
    right_height = np.linalg.norm(quad[2] - quad[1])
    width = max(2, int(round(max(top_width, bottom_width))))
    height = max(2, int(round(max(left_height, right_height))))
    destination = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad, destination)
    image = cv2.warpPerspective(
        frame, transform, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    source_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(source_mask, [np.rint(quad).astype(np.int32)], 255)
    mask = cv2.warpPerspective(
        source_mask, transform, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    )
    if image.size == 0 or cv2.countNonZero(mask) == 0:
        return None
    return SlotCrop(slot_id, image, mask, "perspective")


def _masked_polygon_crop(frame, slot_id, points):
    height, width = frame.shape[:2]
    left = max(0, int(np.floor(np.min(points[:, 0]))))
    top = max(0, int(np.floor(np.min(points[:, 1]))))
    right = min(width, int(np.ceil(np.max(points[:, 0]))) + 1)
    bottom = min(height, int(np.ceil(np.max(points[:, 1]))) + 1)
    if right <= left or bottom <= top:
        raise ValueError(f"{slot_id}: polygon is outside the input frame.")
    local_points = np.rint(points - np.array([left, top])).astype(np.int32)
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 255)
    image = cv2.bitwise_and(frame[top:bottom, left:right],
                            frame[top:bottom, left:right], mask=mask)
    if cv2.countNonZero(mask) == 0:
        raise ValueError(f"{slot_id}: polygon crop contains no pixels.")
    return SlotCrop(slot_id, image, mask, "polygon_mask")


def crop_slot(frame, slot, perspective=True):
    """Crop a `SlotMask` or (slot_id, points) without changing saved geometry."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Input frame must be a 3-channel BGR image.")
    if hasattr(slot, "slot_id"):
        slot_id = str(slot.slot_id)
        points = np.asarray(slot.points, dtype=np.float32)
        if slot.mask.shape != frame.shape[:2]:
            raise ValueError(f"{slot_id}: mask size does not match frame size.")
    else:
        slot_id, points = slot
        slot_id = str(slot_id)
        points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"{slot_id}: expected a polygon with at least 3 points.")
    if not np.isfinite(points).all():
        raise ValueError(f"{slot_id}: polygon contains invalid coordinates.")
    if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= frame.shape[1])
            or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= frame.shape[0])):
        raise ValueError(f"{slot_id}: polygon points are outside the frame.")
    if perspective and len(points) == 4:
        crop = _perspective_crop(frame, slot_id, points)
        if crop is not None:
            return crop
    return _masked_polygon_crop(frame, slot_id, points)


def crop_slots(frame, slots, perspective=True):
    """Return crops in slot order while preserving every slot ID."""
    return [crop_slot(frame, slot, perspective=perspective) for slot in slots]


def save_debug_crops(crops, output_dir, prefix="frame"):
    """Write lossless-enough JPEG previews for human crop inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for crop in crops:
        path = output_dir / f"{prefix}_{crop.slot_id}.jpg"
        if not cv2.imwrite(str(path), crop.image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not save crop: {path}")
        check = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if check is None or check.shape != crop.image.shape:
            raise OSError(f"Saved crop failed image validation: {path}")
        written.append(path)
    return written
