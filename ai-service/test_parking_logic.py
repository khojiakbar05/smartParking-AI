import tempfile
import unittest
from pathlib import Path

import numpy as np

from parking_logic import (
    TemporalOccupancy,
    build_slot_masks,
    load_slots,
    slot_overlap_ratios,
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


if __name__ == "__main__":
    unittest.main()
