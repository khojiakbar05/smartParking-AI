import test from "node:test";
import assert from "node:assert/strict";

class FakeHTMLElement {
  constructor() {
    this.attributes = new Map();
    this.shadowRoot = null;
    this.events = [];
  }

  attachShadow() {
    this.shadowRoot = { innerHTML: "" };
    return this.shadowRoot;
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  setAttribute(name, value) {
    const oldValue = this.getAttribute(name);
    this.attributes.set(name, String(value));
    this.attributeChangedCallback?.(name, oldValue, String(value));
  }

  dispatchEvent(event) {
    this.events.push(event);
    return true;
  }
}

class FakeCustomEvent {
  constructor(type, options) {
    this.type = type;
    Object.assign(this, options);
  }
}

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

  close(code = 1000) {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code });
  }
}

globalThis.HTMLElement = FakeHTMLElement;
globalThis.CustomEvent = FakeCustomEvent;
globalThis.WebSocket = FakeWebSocket;
const definitions = new Map();
globalThis.customElements = {
  define: (name, component) => definitions.set(name, component),
  get: (name) => definitions.get(name),
};

const { SmartParkingHealthMonitor } = await import("./systemHealthMonitor.mjs");

test("health panel displays real heartbeat fields, stale state, and disconnects", (t) => {
  FakeWebSocket.instances = [];
  const monitor = new SmartParkingHealthMonitor();
  monitor.setAttribute("ws-url", "ws://local/ws/parking");
  monitor.connectedCallback();
  t.after(() => monitor.disconnectedCallback());

  assert.equal(FakeWebSocket.instances.length, 1);
  const socket = FakeWebSocket.instances[0];
  socket.open();
  socket.message({ type: "pipeline_status", status: "starting", camera_status: "starting" });
  socket.message({
    type: "heartbeat",
    pipeline_status: "running",
    camera_status: "connected",
    last_inference_at: "2026-11-16T01:02:03+00:00",
    server_timestamp: "2026-11-16T01:02:05+00:00",
  });

  assert.match(monitor.shadowRoot.innerHTML, /Connected/);
  assert.match(monitor.shadowRoot.innerHTML, /connected/);
  assert.match(monitor.shadowRoot.innerHTML, /running/);
  assert.match(monitor.shadowRoot.innerHTML, /Latest inference/);
  assert.match(monitor.shadowRoot.innerHTML, /Last heartbeat received/);

  monitor._lease.client.lastHeartbeatAt = Date.now() - 15_001;
  monitor._lease.client._refreshLiveness();
  assert.match(monitor.shadowRoot.innerHTML, /Stale/);

  socket.close(1006);
  assert.match(monitor.shadowRoot.innerHTML, /Disconnected/);
});

test("health panel forwards occupancy events and never creates an unconfigured socket", (t) => {
  FakeWebSocket.instances = [];
  const missingUrl = new SmartParkingHealthMonitor();
  missingUrl.connectedCallback();
  t.after(() => missingUrl.disconnectedCallback());
  assert.equal(FakeWebSocket.instances.length, 0);
  assert.match(missingUrl.shadowRoot.innerHTML, /Not configured/);

  const monitor = new SmartParkingHealthMonitor();
  monitor.setAttribute("ws-url", "ws://local/occupancy");
  monitor.connectedCallback();
  t.after(() => monitor.disconnectedCallback());
  const socket = FakeWebSocket.instances[0];
  socket.open();
  const parkingStatus = { type: "parking_status", slots: [{ slot_id: "P01" }] };
  socket.message(parkingStatus);
  assert.equal(monitor.events.at(-1).type, "parking-status");
  assert.deepEqual(monitor.events.at(-1).detail, parkingStatus);
});
