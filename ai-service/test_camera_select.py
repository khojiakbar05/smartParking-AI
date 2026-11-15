import unittest
from unittest.mock import patch

from camera_devices import CameraResult
import camera_select


class CameraSelectTests(unittest.TestCase):
    def test_selection_probes_previews_and_saves_only_selected_index(self):
        result = CameraResult(
            index=camera_select.IPHONE_CAMERA_INDEX,
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

        selected_index = camera_select.IPHONE_CAMERA_INDEX
        probe.assert_called_once_with(selected_index)
        preview.assert_called_once_with(selected_index)
        save.assert_called_once_with(selected_index)


if __name__ == "__main__":
    unittest.main()
