"""Protocol tests that do not start YOLO or open the camera."""

from datetime import datetime
from queue import Queue

from fastapi.testclient import TestClient

from api.main import create_app


class IdlePipelineService:
    """Test-only service state; intentionally supplies no parking snapshot."""

    def start(self):
        pass

    def stop(self):
        pass

    def status(self):
        return {
            "pipeline_status": "starting",
            "camera_status": "starting",
            "ready": False,
            "last_update": None,
            "state_changed_at": None,
            "error": None,
        }

    def snapshot(self):
        return None

    def subscribe(self):
        return Queue(maxsize=1), None, "starting", None

    def unsubscribe(self, _subscriber):
        pass


def test_websocket_heartbeat_includes_liveness_only_fields():
    app = create_app(
        pipeline_service=IdlePipelineService(),
        websocket_heartbeat_seconds=0.05,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/parking") as socket:
            initial = socket.receive_json()
            heartbeat = socket.receive_json()
            next_heartbeat = socket.receive_json()

    assert initial["type"] == "pipeline_status"
    assert heartbeat["type"] == "heartbeat"
    assert heartbeat["pipeline_status"] == "starting"
    assert heartbeat["camera_status"] == "starting"
    assert heartbeat["last_inference_at"] is None
    assert "server_timestamp" in heartbeat
    datetime.fromisoformat(heartbeat["server_timestamp"])
    assert "slots" not in heartbeat
    assert "summary" not in heartbeat
    assert "vehicle_count" not in heartbeat
    assert next_heartbeat["type"] == "heartbeat"
    assert "slots" not in next_heartbeat
