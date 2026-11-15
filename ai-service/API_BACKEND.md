# SmartParking FastAPI backend

The API starts the existing `parking_detection.run` pipeline once in a
background thread. That runner loads YOLO once, opens the selected AVFoundation
camera once, and publishes snapshots from the same filtered detections,
polygon overlap logic, and temporal smoothing used by the standalone preview.
The API path is headless; the standalone script still shows its OpenCV window.

`camera_devices.py` holds an advisory process lock for the lifetime of an open
camera handle. This prevents another API worker, standalone script, editor, or
camera utility that uses the shared camera helper from opening the device
concurrently. Run Uvicorn with one worker (`--workers 1`). Shutdown signals the
pipeline, waits for the capture loop, and releases the camera handle.

## Endpoints

- `GET /api/health`: pipeline state, readiness, last result time, and startup
  or runtime error (if any).
- `GET /api/parking/status`: latest actual inference snapshot. Returns HTTP 503
  until the first frame has been processed or if startup failed.
- `GET /api/parking/slots`: saved slot IDs and latest states; slots are
  `UNKNOWN` with null confidence before inference is available.
- `WS /ws/parking`: initial latest snapshot followed by changed snapshots and
  a lightweight heartbeat every five seconds. Heartbeats include pipeline
  state, camera state, and the timestamp of the last processed frame, not
  repeated or invented parking data. Each heartbeat has `type: "heartbeat"`,
  `pipeline_status`, `camera_status`, `last_inference_at`, and
  `server_timestamp` (the legacy `timestamp` field is retained). Occupancy
  snapshots remain change-only. A bounded latest-only queue prevents slow
  clients from blocking inference.

## Dashboard WebSocket liveness helper

This repository does not currently include a React dashboard. The small
framework-agnostic client in `../dashboard/parkingWebSocketClient.mjs` can be
used by a dashboard component. It reports `Connected` after a heartbeat,
`Stale` when no heartbeat arrives for 15 seconds, and `Disconnected` as soon
as the socket closes. It reconnects with a single exponential-backoff timer;
calling `start()` repeatedly is safe, and `stop()` clears timers and closes
the socket.

Example browser integration:

```js
import { ParkingWebSocketClient } from "./parkingWebSocketClient.mjs";

const client = new ParkingWebSocketClient("ws://127.0.0.1:8000/ws/parking", {
  onParkingStatus: (message) => renderParking(message),
  onPipelineStatus: (message) => renderPipeline(message),
  onLivenessChange: ({ status, lastHeartbeatAt }) =>
    renderConnection(status, lastHeartbeatAt),
});
client.start();
// On component teardown: client.stop();
```

The helper only forwards parking data from `parking_status` messages. A
heartbeat updates liveness and never updates or fabricates occupancy state.

Legacy vehicle-overlap statuses do not have a calibrated occupancy
probability, so their `confidence` is `null`. The response labels their source
as `vehicle_overlap`; vehicle detector confidence is not misrepresented as
slot confidence. Existing `UNKNOWN` rules for alignment uncertainty and
threshold disagreement are retained. The current top-down/monitor imagery can
still produce unstable vehicle detections, so API transport success is not a
claim of accurate occupancy.

CORS defaults to `http://localhost:3000` and `http://localhost:5173`. Override
with `SMARTPARKING_CORS_ORIGINS` as a comma-separated list. When using
`python -m api.main`, host and port can be set with `SMARTPARKING_API_HOST` and
`SMARTPARKING_API_PORT`; defaults are `127.0.0.1:8000`. Uvicorn CLI flags
override those values. `SMARTPARKING_WS_HEARTBEAT_SECONDS` sets the WebSocket
heartbeat interval; it defaults to 5 seconds and must be positive.

## Start

```bash
cd ai-service
.venv311/bin/python -m api.main
```

Equivalent Uvicorn command:

```bash
cd ai-service
.venv311/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Test HTTP endpoints:

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/parking/status
curl http://127.0.0.1:8000/api/parking/slots
```

Test the WebSocket using the installed `websockets` package:

```bash
cd ai-service
.venv311/bin/python - <<'PY'
import asyncio
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/parking") as socket:
        print(await socket.recv())

asyncio.run(main())
PY
```

For a real subscription, use a WebSocket client pointed at
`ws://127.0.0.1:8000/ws/parking`. Do not start `parking_detection.py`, the slot
editor, or another camera utility while the API is running; the camera lock
will reject the second opener.
