import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from parking_logic import (
    TemporalOccupancy,
    build_slot_masks,
    load_slots,
    slot_overlap_ratios,
    slot_vehicle_overlaps,
)


class ParkingLogicTests(unittest.TestCase):
    def setUp(self):
        self.slots = [("P01", np.array([
            [10, 10], [50, 10], [50, 50], [10, 50]
        ], dtype=np.float32))]
        self.masks = build_slot_masks(self.slots, (100, 100), (100, 100))

    def test_vehicle_polygon_overlap_is_measured_as_fraction_of_slot(self):
        ratios = slot_overlap_ratios(self.masks, [[10, 10, 30, 50]])
        self.assertGreater(ratios["P01"], 0.45)
        self.assertLess(ratios["P01"], 0.55)

    def test_vehicle_outside_slot_has_no_overlap(self):
        ratios = slot_overlap_ratios(self.masks, [[60, 60, 90, 90]])
        self.assertEqual(ratios["P01"], 0.0)

    def test_slot_coordinates_scale_without_aspect_distortion(self):
        masks = build_slot_masks(self.slots, (100, 100), (50, 50))
        self.assertEqual(masks[0].points.tolist(), [[5, 5], [25, 5], [25, 25], [5, 25]])
        ratios = slot_overlap_ratios(masks, [[5, 5, 15, 25]])
        self.assertGreater(ratios["P01"], 0.45)

    def test_different_aspect_ratio_uses_center_fit_without_stretching(self):
        masks = build_slot_masks(self.slots, (100, 100), (100, 50))
        self.assertEqual(masks[0].points.tolist(), [[30, 5], [50, 5], [50, 25], [30, 25]])

    def test_temporal_majority_suppresses_single_frame_detection(self):
        smoother = TemporalOccupancy(["P01"], window_size=5, occupied_votes=3)
        threshold = 0.20
        self.assertFalse(smoother.update({"P01": 0.8}, threshold)["P01"])
        self.assertFalse(smoother.update({"P01": 0.8}, threshold)["P01"])
        self.assertTrue(smoother.update({"P01": 0.8}, threshold)["P01"])
        self.assertTrue(smoother.update({"P01": 0.0}, threshold)["P01"])
        self.assertTrue(smoother.update({"P01": 0.0}, threshold)["P01"])
        self.assertFalse(smoother.update({"P01": 0.0}, threshold)["P01"])

    def test_load_existing_slot_json_format(self):
        data = '{"frame_width": 100, "frame_height": 100, "slots": [' \
               '{"id": "P01", "points": [[10,10],[50,10],[50,50],[10,50]]}]}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slots.json"
            path.write_text(data, encoding="utf-8")
            width, height, slots = load_slots(path)
        self.assertEqual((width, height), (100, 100))
        self.assertEqual(slots[0][0], "P01")
        self.assertEqual(slots[0][1].shape, (4, 2))

    def test_vehicle_to_slot_debug_reports_pair_and_percent(self):
        overlaps = slot_vehicle_overlaps(self.masks, [[10, 10, 30, 50]])
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["vehicle_index"], 0)
        self.assertEqual(overlaps[0]["slot_id"], "P01")
        self.assertGreater(overlaps[0]["ratio"], 0.45)

    def test_current_slot_file_p03_crossing_is_fixed_only_in_memory(self):
        path = Path(__file__).with_name("parking_slots.json")
        original = path.read_bytes()
        with self.assertWarns(RuntimeWarning):
            _, _, slots = load_slots(path)
        self.assertEqual(path.read_bytes(), original)
        p03 = next(points for slot_id, points in slots if slot_id == "P03")
        self.assertGreater(cv2.contourArea(p03.astype(np.int32)), 10000)

    def test_out_of_bounds_saved_points_are_rejected(self):
        data = '{"frame_width": 100, "frame_height": 100, "slots": [' \
               '{"id": "P01", "points": [[10,10],[110,10],[50,50]]}]}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slots.json"
            path.write_text(data, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "kadr chegarasidan tashqarida"):
                load_slots(path)


if __name__ == "__main__":
    unittest.main()
