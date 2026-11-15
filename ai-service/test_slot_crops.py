"""Tests for polygon-aware parking-slot crops and UNKNOWN classification."""

import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

from parking_logic import build_slot_masks, load_slots
from collect_slot_samples import _save_sample
from slot_classifier import (
    SlotPrediction,
    SlotStatus,
    SlotStatusSmoother,
    UnknownSlotClassifier,
    fuse_classifier_and_vehicle,
)
from slot_crops import crop_slot, crop_slots


class SlotCropTests(unittest.TestCase):
    def test_current_and_historical_slot_layouts_crop_preserve_ids(self):
        service_dir = Path(__file__).parent
        layouts = [service_dir / "parking_slots.json"]
        historical = service_dir / "debug" / "slots_16_historical.json"
        if historical.exists():
            layouts.append(historical)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (20, 40, 60)
        for layout in layouts:
            with self.subTest(layout=layout.name):
                width, height, slots = load_slots(layout)
                masks = build_slot_masks(slots, (width, height), (1280, 720))
                crops = crop_slots(frame, masks)
                self.assertEqual([item.slot_id for item in crops],
                                 [item.slot_id for item in masks])
                self.assertTrue(all(item.image.size and item.mask.size
                                    for item in crops))
                self.assertTrue(all(item.method in ("perspective", "polygon_mask")
                                    for item in crops))

    def test_non_quad_polygon_uses_mask_and_blacks_out_corners(self):
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        crop = crop_slot(frame, ("P-test", [(10, 10), (80, 10), (50, 50),
                                             (80, 80), (10, 80)]), perspective=False)
        self.assertEqual(crop.slot_id, "P-test")
        self.assertEqual(crop.method, "polygon_mask")
        self.assertTrue(np.all(crop.image[10, 69] == 0))
        self.assertTrue(np.all(crop.image[20, 20] == 200))

    def test_rejects_out_of_frame_polygon(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            crop_slot(frame, ("bad", [(-1, 1), (20, 1), (20, 20)]))

    def test_unknown_defaults_and_vehicle_fusion_respects_confident_classifier(self):
        crop = crop_slot(np.zeros((40, 40, 3), dtype=np.uint8),
                         ("P01", [(2, 2), (30, 2), (30, 30), (2, 30)]))
        unknown = UnknownSlotClassifier().classify_batch([crop])["P01"]
        self.assertEqual(unknown.status, SlotStatus.UNKNOWN)
        supported = fuse_classifier_and_vehicle(
            unknown, [{"confidence": 0.9, "ratio": 0.3}]
        )
        self.assertEqual(supported.status, SlotStatus.OCCUPIED)
        vacant = SlotPrediction("P01", SlotStatus.VACANT, 0.91, unknown.timestamp)
        protected = fuse_classifier_and_vehicle(
            vacant, [{"confidence": 0.99, "ratio": 0.9}]
        )
        self.assertEqual(protected.status, SlotStatus.VACANT)

    def test_smoother_keeps_insufficient_evidence_unknown(self):
        smoother = SlotStatusSmoother(["P01"], window_size=5, required_votes=3)
        unknown = SlotPrediction("P01", SlotStatus.UNKNOWN, 0.2, "t")
        self.assertEqual(smoother.update({"P01": unknown})["P01"].status,
                         SlotStatus.UNKNOWN)
        vacant = SlotPrediction("P01", SlotStatus.VACANT, 0.9, "t")
        for _ in range(2):
            result = smoother.update({"P01": vacant})["P01"]
        self.assertEqual(result.status, SlotStatus.UNKNOWN)
        result = smoother.update({"P01": vacant})["P01"]
        self.assertEqual(result.status, SlotStatus.VACANT)

    def test_manual_collector_rejects_duplicate_and_writes_valid_jpeg(self):
        crop = crop_slot(np.full((40, 40, 3), 120, dtype=np.uint8),
                         ("P01", [(2, 2), (30, 2), (30, 30), (2, 30)]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            known = {}
            first = _save_sample(crop, "occupied", root, "P01", (1280, 720), known)
            self.assertIsNotNone(first)
            saved = cv2.imread(str(first), cv2.IMREAD_COLOR)
            self.assertIsNotNone(saved)
            second = _save_sample(crop, "occupied", root, "P01", (1280, 720), known)
            self.assertIsNone(second)
            self.assertEqual(len(list((root / "occupied").glob("*.jpg"))), 1)


if __name__ == "__main__":
    unittest.main()
