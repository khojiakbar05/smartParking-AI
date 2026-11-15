import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from camera_devices import load_selection, open_camera, probe_camera, save_selection


class FakeCapture:
    def __init__(self, frames, opened=True, fps=0):
        self.frames = iter(frames)
        self.opened = opened
        self.fps = fps
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return next(self.frames, (False, None))

    def get(self, _property):
        return self.fps

    def release(self):
        self.released = True


class CameraDeviceTests(unittest.TestCase):
    def test_requires_three_consecutive_valid_frames_and_releases(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        capture = FakeCapture([
            (True, frame), (False, None),
            (True, frame), (True, frame), (True, frame),
        ], fps=30)
        result = probe_camera(7, attempts=5, capture_factory=lambda *_: capture)
        self.assertTrue(result.opened)
        self.assertTrue(result.frame_read_success)
        self.assertTrue(result.stable)
        self.assertEqual(result.resolution, "640x480")
        self.assertEqual(result.fps, 30.0)
        self.assertTrue(capture.released)

    def test_empty_or_wrong_sized_frames_do_not_count_as_stable(self):
        first = np.zeros((480, 640, 3), dtype=np.uint8)
        second = np.zeros((240, 320, 3), dtype=np.uint8)
        capture = FakeCapture([
            (True, first), (True, second), (True, second),
            (False, None), (True, second),
        ])
        result = probe_camera(3, attempts=5, capture_factory=lambda *_: capture)
        self.assertTrue(result.frame_read_success)
        self.assertFalse(result.stable)
        self.assertTrue(capture.released)

    def test_closed_camera_reports_not_open_and_releases(self):
        capture = FakeCapture([], opened=False)
        result = probe_camera(5, capture_factory=lambda *_: capture)
        self.assertFalse(result.opened)
        self.assertFalse(result.frame_read_success)
        self.assertTrue(capture.released)

    def test_saved_selection_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "camera.json"
            save_selection(6, config)
            self.assertEqual(load_selection(config), 6)

    def test_camera_session_lock_rejects_second_process_session_until_release(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "camera.lock"
            first_capture = FakeCapture([], opened=True)
            second_capture = FakeCapture([], opened=True)
            with patch("camera_devices.CAMERA_SESSION_LOCK_PATH", lock_path):
                first = open_camera(0, capture_factory=lambda *_: first_capture)
                with self.assertRaisesRegex(RuntimeError, "boshqa SmartParking"):
                    open_camera(0, capture_factory=lambda *_: second_capture)
                self.assertFalse(second_capture.released)
                first.release()
                second = open_camera(0, capture_factory=lambda *_: second_capture)
                second.release()
            self.assertTrue(first_capture.released)
            self.assertTrue(second_capture.released)


if __name__ == "__main__":
    unittest.main()
