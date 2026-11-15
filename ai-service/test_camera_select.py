import unittest
from unittest.mock import patch

from camera_devices import CameraResult
import camera_select


class CameraSelectTests(unittest.TestCase):
    def test_selection_probes_previews_and_saves_only_index_one(self):
        result = CameraResult(
            index=1,
            opened=True,
            frame_read_success=True,
            stable=True,
            width=1280,
            height=720,
        )
        with (
            patch("camera_select.probe_camera", return_value=result) as probe,
            patch("camera_select.preview", return_value=True) as preview,
            patch("camera_select.save_selection") as save,
        ):
            self.assertEqual(camera_select.main(), 0)

        probe.assert_called_once_with(1)
        preview.assert_called_once_with(1)
        save.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
