import test from "node:test";
import assert from "node:assert/strict";

import {
  acquireParkingWebSocketClient,
  LIVENESS_STATES,
  ParkingWebSocketClient,
} from "./parkingWebSocketClient.mjs";

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    FakeWebSocket.instances.push(this);
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.({});
  }

  message(value) {
    this.onmessage?.({ data: JSON.stringify(value) });
  }

  close(code = 1000, reason = "closed") {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code, reason });
  }
}

async function waitFor(predicate, timeoutMs = 500) {
  const started = Date.now();
  while (!predicate()) {
    if (Date.now() - started > timeoutMs) throw new Error("Timed out waiting for condition.");
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
}

test("client updates liveness only from heartbeats and marks a missed heartbeat stale", async (t) => {
  FakeWebSocket.instances = [];
  const liveness = [];
  const parkingUpdates = [];
  const client = new ParkingWebSocketClient("ws://local/ws/parking", {
    WebSocketImpl: FakeWebSocket,
    onLivenessChange: (event) => liveness.push(event),
    onParkingStatus: (event) => parkingUpdates.push(event),
    heartbeatTimeoutMs: 50,
    livenessPollMs: 5,
    reconnectMinMs: 100,
    reconnectMaxMs: 100,
    logger: {},
  });
  t.after(() => client.stop());

  client.start();
  client.start();
  assert.equal(FakeWebSocket.instances.length, 1, "start is idempotent");
  const socket = FakeWebSocket.instances[0];
  socket.open();
  assert.equal(client.liveness, LIVENESS_STATES.CONNECTING);

  socket.message({ type: "parking_status", summary: { vehicle_count: 2 } });
  assert.equal(parkingUpdates.length, 1);
  assert.equal(client.liveness, LIVENESS_STATES.CONNECTING,
    "parking updates do not refresh heartbeat liveness");

  socket.message({
    type: "heartbeat",
    pipeline_status: "running",
    camera_status: "connected",
    last_inference_at: "2026-11-16T00:00:00+00:00",
    server_timestamp: "2026-11-16T00:00:01+00:00",
  });
  assert.equal(client.liveness, LIVENESS_STATES.CONNECTED);
  assert.ok(liveness.at(-1).lastHeartbeatAt);
  assert.equal(liveness.at(-1).heartbeat.camera_status, "connected");

  await waitFor(() => client.liveness === LIVENESS_STATES.STALE);
  assert.equal(client.liveness, LIVENESS_STATES.STALE);
});

test("client reports close and reconnects only once, then stop cancels reconnect", async () => {
  FakeWebSocket.instances = [];
  const states = [];
  const client = new ParkingWebSocketClient("ws://local/ws/parking", {
    WebSocketImpl: FakeWebSocket,
    onLivenessChange: ({ status }) => states.push(status),
    heartbeatTimeoutMs: 1_000,
    livenessPollMs: 10,
    reconnectMinMs: 10,
    reconnectMaxMs: 20,
    logger: {},
  });

  client.start();
  const first = FakeWebSocket.instances[0];
  first.open();
  first.close(1006, "network loss");
  assert.equal(client.liveness, LIVENESS_STATES.DISCONNECTED);
  await waitFor(() => FakeWebSocket.instances.length === 2);
  client.start();
  client.start();
  assert.equal(FakeWebSocket.instances.length, 2, "reconnect creates only one socket");

  client.stop();
  assert.equal(client.liveness, LIVENESS_STATES.DISCONNECTED);
  await new Promise((resolve) => setTimeout(resolve, 30));
  assert.equal(FakeWebSocket.instances.length, 2, "stop cancels pending reconnect");
  assert.ok(states.includes(LIVENESS_STATES.DISCONNECTED));
});

test("same-URL dashboard subscribers share one socket and release it by reference count", () => {
  FakeWebSocket.instances = [];
  const firstUpdates = [];
  const secondUpdates = [];
  const options = {
    WebSocketImpl: FakeWebSocket,
    logger: {},
  };
  const first = acquireParkingWebSocketClient("ws://local/shared", {
    onParkingStatus: (message) => firstUpdates.push(message),
  }, options);
  const second = acquireParkingWebSocketClient("ws://local/shared", {
    onParkingStatus: (message) => secondUpdates.push(message),
  }, options);

  assert.equal(FakeWebSocket.instances.length, 1);
  assert.equal(first.client, second.client);
  const socket = FakeWebSocket.instances[0];
  socket.open();
  socket.message({ type: "parking_status", summary: { total_slots: 4 } });
  assert.equal(firstUpdates.length, 1);
  assert.equal(secondUpdates.length, 1);

  first.release();
  assert.equal(socket.readyState, FakeWebSocket.OPEN);
  second.release();
  assert.equal(socket.readyState, FakeWebSocket.CLOSED);
});
