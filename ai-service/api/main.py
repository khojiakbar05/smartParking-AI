"""HTTP and WebSocket API backed by one existing parking pipeline instance."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import math
import os
from queue import Empty

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.service import ParkingPipelineService
from parking_logic import load_slots
from parking_detection import SLOTS_PATH

logger = logging.getLogger(__name__)


def _cors_origins():
    configured = os.getenv("SMARTPARKING_CORS_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://localhost:5173"]


def _heartbeat_interval(configured=None):
    value = (os.getenv("SMARTPARKING_WS_HEARTBEAT_SECONDS", "5")
             if configured is None else configured)
    try:
        interval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("WebSocket heartbeat interval must be a positive number.") from exc
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("WebSocket heartbeat interval must be a positive number.")
    return interval


def create_app(pipeline_service=None, websocket_heartbeat_seconds=None):
    service = pipeline_service or ParkingPipelineService()
    heartbeat_seconds = _heartbeat_interval(websocket_heartbeat_seconds)

    @asynccontextmanager
    async def lifespan(_app):
        service.start()
        yield
        await asyncio.to_thread(service.stop)

    app = FastAPI(
        title="SmartParking AI API",
        version="1.0.0",
        description="Live parking data from the existing camera and AI pipeline.",
        lifespan=lifespan,
    )
    app.state.parking_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        state = service.status()
        return {"success": True, **state}

    @app.get("/api/parking/status")
    def parking_status():
        snapshot = service.snapshot()
        state = service.status()
        if snapshot is None or state["pipeline_status"] != "running":
            detail = {
                "success": False,
                "error": state["error"] or "No camera inference result is available yet.",
                "pipeline_status": state["pipeline_status"],
            }
            raise HTTPException(status_code=503, detail=detail)
        return snapshot

    @app.get("/api/parking/slots")
    def parking_slots():
        try:
            _width, _height, definitions = load_slots(SLOTS_PATH)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"success": False, "error": f"Could not read slot config: {exc}"},
            ) from exc
        pipeline_state = service.status()
        snapshot = (service.snapshot()
                    if pipeline_state["pipeline_status"] == "running" else None)
        by_id = {item["slot_id"]: item for item in (snapshot or {}).get("slots", [])}
        slots = [
            by_id.get(slot_id, {
                "slot_id": slot_id,
                "status": "UNKNOWN",
                "confidence": None,
            })
            for slot_id, _points in definitions
        ]
        return {
            "success": True,
            "timestamp": (snapshot or {}).get("timestamp"),
            "pipeline_status": pipeline_state["pipeline_status"],
            "total_slots": len(slots),
            "slots": slots,
        }

    @app.websocket("/ws/parking")
    async def parking_updates(websocket: WebSocket):
        await websocket.accept()
        subscriber, snapshot, state, error = service.subscribe()
        connection_status = service.status()
        receive_task = None
        event_task = None
        try:
            if snapshot is not None and state == "running":
                await websocket.send_json({"type": "parking_status", **snapshot})
            else:
                await websocket.send_json({
                    "type": "pipeline_status",
                    "success": False,
                    "status": state,
                    "camera_status": connection_status["camera_status"],
                    "timestamp": None,
                    "error": error or "Waiting for the first camera inference result.",
                })
            logger.info("Parking WebSocket client connected")
            loop = asyncio.get_running_loop()
            receive_task = asyncio.create_task(websocket.receive())
            next_heartbeat = loop.time() + heartbeat_seconds
            while True:
                queue_timeout = min(1.0, max(0.01, next_heartbeat - loop.time()))
                event_task = asyncio.create_task(
                    asyncio.to_thread(subscriber.get, True, queue_timeout)
                )
                done, _pending = await asyncio.wait(
                    {receive_task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receive_task in done:
                    message = receive_task.result()
                    if message.get("type") == "websocket.disconnect":
                        break
                    receive_task = asyncio.create_task(websocket.receive())
                if event_task in done:
                    try:
                        event = event_task.result()
                    except Empty:
                        event = None
                    if event is not None:
                        await websocket.send_json(event)
                    event_task = None
                else:
                    event_task.cancel()

                now = loop.time()
                if now >= next_heartbeat:
                    state = service.status()
                    server_timestamp = datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    )
                    await websocket.send_json({
                        "type": "heartbeat",
                        "success": state["pipeline_status"] == "running",
                        "timestamp": server_timestamp,
                        "server_timestamp": server_timestamp,
                        "pipeline_status": state["pipeline_status"],
                        "camera_status": state["camera_status"],
                        "ready": state["ready"],
                        "last_inference_at": state["last_update"],
                    })
                    logger.debug(
                        "Sent parking heartbeat: pipeline=%s camera=%s last_inference=%s",
                        state["pipeline_status"], state["camera_status"],
                        state["last_update"],
                    )
                    next_heartbeat = now + heartbeat_seconds
        except WebSocketDisconnect:
            logger.info("Parking WebSocket client disconnected")
        finally:
            if receive_task is not None:
                receive_task.cancel()
            if event_task is not None:
                event_task.cancel()
            service.unsubscribe(subscriber)
            if websocket.client_state.name != "DISCONNECTED":
                logger.info("Parking WebSocket client connection ended")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SMARTPARKING_API_HOST", "127.0.0.1")
    port = int(os.getenv("SMARTPARKING_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, workers=1)
