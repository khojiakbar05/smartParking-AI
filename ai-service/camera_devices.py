"""Camera discovery and selection helpers for macOS AVFoundation."""

import json
import fcntl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
CONFIG_PATH = Path(__file__).with_name(".camera_selection.json")
CAMERA_SESSION_LOCK_PATH = Path(__file__).with_name(".camera_session.lock")
DEFAULT_MAX_INDEX = 9
STABLE_READS_REQUIRED = 3


def _acquire_camera_session_lock():
    """Prevent another process from opening the selected AVFoundation camera."""
    lock_file = CAMERA_SESSION_LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(
            "Kamera boshqa SmartParking jarayoni tomonidan ishlatilmoqda. "
            "Avval o'sha preview yoki serverni to'xtating."
        ) from exc
    return lock_file


def _release_camera_session_lock(lock_file):
    if lock_file is None:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


class _LockedCapture:
    """Delegate VideoCapture methods and hold the process lock until release."""

    def __init__(self, capture, lock_file):
        self._capture = capture
        self._lock_file = lock_file
        self._released = False

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            self._capture.release()
        finally:
            _release_camera_session_lock(self._lock_file)


@dataclass
class CameraResult:
    index: int
    opened: bool
    frame_read_success: bool
    stable: bool
    successful_reads: int = 0
    total_reads: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    error: Optional[str] = None

    @property
    def resolution(self):
        if self.width is None or self.height is None:
            return "N/A"
        return f"{self.width}x{self.height}"


def probe_camera(index, attempts=8, stable_reads=STABLE_READS_REQUIRED,
                 capture_factory=None):
    """Open one index and verify consecutive, non-empty, consistent frames."""
    factory = capture_factory or cv2.VideoCapture
    result = CameraResult(index=index, opened=False, frame_read_success=False,
                          stable=False)
    capture = None
    stable_run = 0
    stable_size = None
    first_frame_time = None
    last_frame_time = None
    session_lock = None

    try:
        session_lock = _acquire_camera_session_lock()
        capture = factory(index, CAMERA_BACKEND)
        result.opened = bool(capture.isOpened())
        if not result.opened:
            return result

        reported_fps = capture.get(cv2.CAP_PROP_FPS)
        for _ in range(attempts):
            result.total_reads += 1
            ok, frame = capture.read()
            now = time.monotonic()
            if not ok or frame is None or getattr(frame, "ndim", 0) < 2:
                stable_run = 0
                stable_size = None
                continue

            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                stable_run = 0
                stable_size = None
                continue

            result.successful_reads += 1
            result.frame_read_success = True
            size = (width, height)
            if stable_size == size:
                stable_run += 1
            else:
                stable_size = size
                stable_run = 1
            if stable_run >= stable_reads:
                result.stable = True
                result.width, result.height = size
            if first_frame_time is None:
                first_frame_time = now
            last_frame_time = now

        if result.stable and reported_fps and reported_fps > 0:
            result.fps = float(reported_fps)
        elif (result.stable and result.successful_reads > 1 and
              last_frame_time > first_frame_time):
            result.fps = (result.successful_reads - 1) / (
                last_frame_time - first_frame_time
            )
    except Exception as exc:  # AVFoundation may reject or lose a device.
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        _release_camera_session_lock(session_lock)
    return result


def discover_cameras(max_index=DEFAULT_MAX_INDEX, **probe_options):
    """Probe indexes from zero through max_index; indexes are not device IDs."""
    return [probe_camera(index, **probe_options)
            for index in range(max_index + 1)]


def open_camera(index, capture_factory=None):
    """Open a selected camera using AVFoundation, releasing failed handles."""
    factory = capture_factory or cv2.VideoCapture
    session_lock = _acquire_camera_session_lock()
    capture = None
    try:
        capture = factory(int(index), CAMERA_BACKEND)
        if not capture.isOpened():
            raise RuntimeError(f"Kamera indeksi {index} ochilmadi.")
        return _LockedCapture(capture, session_lock)
    except Exception as exc:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        _release_camera_session_lock(session_lock)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Kamera indeksi {index} ochishda xato: {exc}") from exc


def save_selection(index, path=CONFIG_PATH):
    Path(path).write_text(json.dumps({"index": int(index)}, indent=2) + "\n",
                          encoding="utf-8")


def load_selection(path=CONFIG_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = data["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("index manfiy bo'lmagan butun son bo'lishi kerak")
        return index
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Kamera tanlovi fayli noto'g'ri: {exc}") from exc


def open_selected_camera():
    """Open the saved choice; prompt users to run camera_select.py if absent."""
    index = load_selection()
    if index is None:
        raise RuntimeError(
            "Kamera hali tanlanmagan. Avval `python camera_select.py` "
            "buyrug'ini ishga tushiring."
        )
    return open_camera(index)


def print_camera_result(result):
    measured_fps = f"{result.fps:.1f}" if result.fps else "o'lchanmadi"
    read_status = (f"{result.successful_reads}/{result.total_reads} kadr; "
                   f"barqaror: {'HA' if result.stable else 'YO‘Q'}")
    print(f"Indeks {result.index}: {result.resolution}, FPS {measured_fps}, "
          f"kadr o'qish: {read_status}, ochildi: "
          f"{'HA' if result.opened else 'YO‘Q'}")
    if result.error:
        print(f"  Xato: {result.error}")
