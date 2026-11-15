"""Thread-safe bridge between the camera pipeline and API clients."""

import copy
from datetime import datetime, timezone
import logging
from queue import Empty, Queue
import threading

from parking_detection import run as run_parking_pipeline

logger = logging.getLogger(__name__)


class ParkingPipelineService:
    """Own one background pipeline and publish only its real observations."""

    def __init__(self, runner=None):
        self._runner = runner or run_parking_pipeline
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._subscribers = set()
        self._snapshot = None
        self._last_signature = None
        self._state = "stopped"
        self._camera_status = "disconnected"
        self._error = None
        self._state_changed_at = None

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event = threading.Event()
            self._state = "starting"
            self._camera_status = "starting"
            self._error = None
            self._state_changed_at = self._now()
            self._thread = threading.Thread(
                target=self._run,
                name="smartparking-api-pipeline",
                daemon=True,
            )
            self._thread.start()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _run(self):
        try:
            result = self._runner(
                stop_event=self._stop_event,
                on_update=self.publish,
                on_state=self._set_state,
                on_camera_state=self._set_camera_status,
                preview=False,
            )
            with self._lock:
                state = self._state
            if result and state != "error":
                self._set_state("error", f"Parking pipeline exited with code {result}.")
        except Exception as exc:
            logger.exception("Parking pipeline worker failed")
            self._set_state("error", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _signature(snapshot):
        """Ignore per-frame timestamps and insignificant confidence jitter."""
        return (
            tuple(sorted(snapshot["summary"].items())),
            tuple(
                (slot["slot_id"], slot["status"],
                 None if slot["confidence"] is None else round(slot["confidence"], 2))
                for slot in snapshot["slots"]
            ),
            snapshot.get("status_source"),
        )

    def publish(self, snapshot):
        with self._lock:
            self._snapshot = copy.deepcopy(snapshot)
            signature = self._signature(snapshot)
            if signature == self._last_signature:
                return
            self._last_signature = signature
            event = copy.deepcopy(snapshot)
            event["type"] = "parking_status"
            for subscriber in tuple(self._subscribers):
                if subscriber.full():
                    try:
                        subscriber.get_nowait()
                    except Empty:
                        pass
                try:
                    subscriber.put_nowait(event)
                except Exception:
                    # A slow client must never stall inference or other clients.
                    continue

    def _set_state(self, state, error=None):
        with self._lock:
            previous_state = self._state
            self._state = state
            self._error = error
            self._state_changed_at = self._now()
            if previous_state != state:
                logger.info("Parking pipeline status: %s -> %s", previous_state, state)
            if state == "error":
                event = {
                    "type": "pipeline_status",
                    "success": False,
                    "status": state,
                    "timestamp": self._state_changed_at,
                    "error": error,
                }
                for subscriber in tuple(self._subscribers):
                    if subscriber.full():
                        try:
                            subscriber.get_nowait()
                        except Empty:
                            pass
                    try:
                        subscriber.put_nowait(event)
                    except Exception:
                        continue

    def _set_camera_status(self, camera_status):
        with self._lock:
            if self._camera_status == camera_status:
                return
            self._camera_status = camera_status
            logger.info("Parking camera status: %s", camera_status)

    def stop(self, timeout=15.0):
        with self._lock:
            thread = self._thread
            if thread is None:
                self._state = "stopped"
                self._camera_status = "disconnected"
                self._state_changed_at = self._now()
                return
            self._state = "stopping"
            self._state_changed_at = self._now()
            self._stop_event.set()
        if thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            if thread.is_alive():
                self._state = "error"
                self._error = "Pipeline shutdown timed out; camera release is pending."
                self._state_changed_at = self._now()
            elif self._state != "error":
                self._state = "stopped"
                self._state_changed_at = self._now()

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def status(self):
        with self._lock:
            return {
                "pipeline_status": self._state,
                "camera_status": self._camera_status,
                "ready": self._snapshot is not None and self._state == "running",
                "last_update": (self._snapshot or {}).get("timestamp"),
                "state_changed_at": self._state_changed_at,
                "error": self._error,
            }

    def subscribe(self):
        subscriber = Queue(maxsize=1)
        with self._lock:
            self._subscribers.add(subscriber)
            snapshot = copy.deepcopy(self._snapshot)
            state = self._state
            error = self._error
        return subscriber, snapshot, state, error

    def unsubscribe(self, subscriber):
        with self._lock:
            self._subscribers.discard(subscriber)

