import unittest
from types import SimpleNamespace

import numpy as np

from parking_detection import (
    detection_signature,
    filter_detections,
    verify_model_weights,
)


class YoloDebugTests(unittest.TestCase):
    def test_python_vehicle_filter_keeps_only_coco_vehicle_ids(self):
        detections = (
            np.array([[0, 0, 20, 20], [1, 1, 30, 30], [2, 2, 40, 40]], dtype=float),
            np.array([0.91, 0.82, 0.73]),
            np.array([2, 0, 7]),
            None,
        )
        filtered = filter_detections(detections, [2, 3, 5, 7])
        self.assertEqual(filtered[2].tolist(), [2, 7])
        self.assertEqual(filtered[1].tolist(), [0.91, 0.73])

    def test_detection_signature_is_order_independent(self):
        first = (
            np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
            np.array([0.8, 0.9]), np.array([2, 7]), None,
        )
        second = (
            first[0][::-1], first[1][::-1], first[2][::-1], None,
        )
        self.assertEqual(detection_signature(first), detection_signature(second))

    def test_weight_guard_checks_expected_coco_labels(self):
        names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        model = SimpleNamespace(
            task="detect", names=names, model=SimpleNamespace(nc=80)
        )
        verify_model_weights(model)
        model.names = {**names, 2: "bird"}
        with self.assertRaises(RuntimeError):
            verify_model_weights(model)


if __name__ == "__main__":
    unittest.main()
