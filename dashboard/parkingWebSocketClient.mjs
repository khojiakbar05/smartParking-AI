/** Small browser WebSocket client for the SmartParking dashboard. */

export const LIVENESS_STATES = Object.freeze({
  CONNECTING: "Connecting",
  CONNECTED: "Connected",
  STALE: "Stale",
  DISCONNECTED: "Disconnected",
});

const sharedClients = new Map();

/** Share one socket per URL between dashboard widgets and release it by refcount. */
export function acquireParkingWebSocketClient(url, handlers = {}, options = {}) {
  let entry = sharedClients.get(url);
  if (!entry) {
    entry = {
      client: new ParkingWebSocketClient(url, options),
      references: 0,
    };
    sharedClients.set(url, entry);
  }

  entry.references += 1;
  const unsubscribe = entry.client.subscribe(handlers);
  entry.client.start();
  let released = false;

  return {
    client: entry.client,
    release() {
      if (released) return;
      released = true;
      unsubscribe();
      entry.references -= 1;
      if (entry.references === 0) {
        entry.client.stop();
        if (sharedClients.get(url) === entry) sharedClients.delete(url);
      }
    },
  };
}

export class ParkingWebSocketClient {
  constructor(url, {
    WebSocketImpl = globalThis.WebSocket,
    onParkingStatus = () => {},
    onLivenessChange = () => {},
    onPipelineStatus = () => {},
    onHeartbeat = () => {},
    heartbeatTimeoutMs = 15_000,
    livenessPollMs = 1_000,
    reconnectMinMs = 500,
    reconnectMaxMs = 10_000,
    logger = console,
  } = {}) {
    if (!url) throw new TypeError("A WebSocket URL is required.");
    if (typeof WebSocketImpl !== "function") {
      throw new TypeError("A WebSocket implementation is required.");
    }
    if (!(heartbeatTimeoutMs > 0) || !(livenessPollMs > 0)) {
      throw new RangeError("Heartbeat timeout and poll interval must be positive.");
    }
    if (!(reconnectMinMs > 0) || reconnectMaxMs < reconnectMinMs) {
      throw new RangeError("Reconnect delay range is invalid.");
    }

    this.url = url;
    this.WebSocketImpl = WebSocketImpl;
    this.onParkingStatus = onParkingStatus;
    this.onLivenessChange = onLivenessChange;
    this.onPipelineStatus = onPipelineStatus;
    this.onHeartbeat = onHeartbeat;
    this.heartbeatTimeoutMs = heartbeatTimeoutMs;
    this.livenessPollMs = livenessPollMs;
    this.reconnectMinMs = reconnectMinMs;
    this.reconnectMaxMs = reconnectMaxMs;
    this.logger = logger;

    this.started = false;
    this.socket = null;
    this.openedAt = null;
    this.lastHeartbeatAt = null;
    this.lastHeartbeat = null;
    this.liveness = LIVENESS_STATES.DISCONNECTED;
    this.pollTimer = null;
    this.reconnectTimer = null;
    this.reconnectAttempt = 0;
    this.listeners = new Set();
  }

  subscribe(handlers = {}) {
    this.listeners.add(handlers);
    if (typeof handlers.onLivenessChange === "function") {
      this._call(handlers.onLivenessChange, {
        status: this.liveness,
        lastHeartbeatAt: this.lastHeartbeatAt,
        heartbeat: this.lastHeartbeat,
      }, "liveness callback");
    }
    return () => this.listeners.delete(handlers);
  }

  start() {
    if (this.started) return;
    this.started = true;
    this._setLiveness(LIVENESS_STATES.CONNECTING, true);
    this.pollTimer = setInterval(() => this._refreshLiveness(), this.livenessPollMs);
    this._connect();
  }

  stop() {
    if (!this.started && !this.socket && this.pollTimer === null
        && this.reconnectTimer === null) return;

    this.started = false;
    if (this.pollTimer !== null) clearInterval(this.pollTimer);
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
    this.pollTimer = null;
    this.reconnectTimer = null;

    const socket = this.socket;
    this.socket = null;
    this.openedAt = null;
    this.lastHeartbeatAt = null;
    this.lastHeartbeat = null;
    if (socket && socket.readyState < this._socketState("CLOSING", 2)) {
      try {
        socket.close(1000, "Dashboard client stopped");
      } catch (error) {
        this.logger?.debug?.("WebSocket close during stop failed", error);
      }
    }
    this._setLiveness(LIVENESS_STATES.DISCONNECTED, true);
  }

  _socketState(name, fallback) {
    return this.WebSocketImpl[name] ?? fallback;
  }

  _connect() {
    if (!this.started || this.socket) return;
    let socket;
    try {
      socket = new this.WebSocketImpl(this.url);
    } catch (error) {
      this.logger?.warn?.("Could not create parking WebSocket", error);
      this._setLiveness(LIVENESS_STATES.DISCONNECTED, true);
      this._scheduleReconnect();
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      if (!this.started || this.socket !== socket) {
        socket.close();
        return;
      }
      this.openedAt = Date.now();
      this._setLiveness(LIVENESS_STATES.CONNECTING, true);
    };

    socket.onmessage = (event) => this._handleMessage(event, socket);

    socket.onerror = (error) => {
      this.logger?.warn?.("Parking WebSocket reported an error", error);
    };

    socket.onclose = (event) => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.openedAt = null;
      this.lastHeartbeatAt = null;
      this.lastHeartbeat = null;
      this._setLiveness(LIVENESS_STATES.DISCONNECTED, true);
      this.logger?.info?.("Parking WebSocket closed", event?.code);
      this._scheduleReconnect();
    };
  }

  _handleMessage(event, socket) {
    if (this.socket !== socket || !this.started) return;
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (error) {
      this.logger?.warn?.("Ignoring invalid parking WebSocket JSON", error);
      return;
    }

    if (message?.type === "heartbeat") {
      this.lastHeartbeatAt = Date.now();
      this.lastHeartbeat = message;
      this.reconnectAttempt = 0;
      this._call(this.onHeartbeat, message, "heartbeat callback");
      this._notify("onHeartbeat", message, "heartbeat callback");
      this._setLiveness(LIVENESS_STATES.CONNECTED, true);
      return;
    }
    if (message?.type === "parking_status") {
      this._call(this.onParkingStatus, message, "parking-status callback");
      this._notify("onParkingStatus", message, "parking-status callback");
      return;
    }
    if (message?.type === "pipeline_status") {
      this._call(this.onPipelineStatus, message, "pipeline-status callback");
      this._notify("onPipelineStatus", message, "pipeline-status callback");
    }
  }

  _refreshLiveness() {
    if (!this.started || !this.socket) return;
    const openState = this._socketState("OPEN", 1);
    if (this.socket.readyState !== openState) return;

    const referenceTime = this.lastHeartbeatAt ?? this.openedAt;
    if (referenceTime === null) {
      this._setLiveness(LIVENESS_STATES.CONNECTING);
    } else if (Date.now() - referenceTime >= this.heartbeatTimeoutMs) {
      this._setLiveness(LIVENESS_STATES.STALE);
    } else {
      this._setLiveness(LIVENESS_STATES.CONNECTED);
    }
  }

  _scheduleReconnect() {
    if (!this.started || this.reconnectTimer !== null) return;
    const delay = Math.min(
      this.reconnectMaxMs,
      this.reconnectMinMs * (2 ** this.reconnectAttempt),
    );
    this.reconnectAttempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this._connect();
    }, delay);
  }

  _setLiveness(status, force = false) {
    const changed = this.liveness !== status;
    this.liveness = status;
    if (!changed && !force) return;
    this._call(this.onLivenessChange, {
      status,
      lastHeartbeatAt: this.lastHeartbeatAt,
      heartbeat: this.lastHeartbeat,
    }, "liveness callback");
    this._notify("onLivenessChange", {
      status,
      lastHeartbeatAt: this.lastHeartbeatAt,
      heartbeat: this.lastHeartbeat,
    }, "liveness callback");
  }

  _notify(name, value, label) {
    for (const listener of this.listeners) {
      if (typeof listener?.[name] === "function") {
        this._call(listener[name], value, label);
      }
    }
  }

  _call(callback, value, label) {
    try {
      callback(value);
    } catch (error) {
      this.logger?.error?.(`Parking WebSocket ${label} failed`, error);
    }
  }
}
