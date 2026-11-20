import { acquireParkingWebSocketClient } from "./parkingWebSocketClient.mjs";

const HTMLElementBase = globalThis.HTMLElement ?? class {};

/**
 * Framework-independent system health panel. Add `ws-url` to point it at the
 * existing API and listen for its bubbling `parking-status` events to retain
 * the dashboard's current occupancy rendering.
 */
export class SmartParkingHealthMonitor extends HTMLElementBase {
  static get observedAttributes() {
    return ["ws-url"];
  }

  constructor() {
    super();
    this._lease = null;
    this._connected = false;
    this._health = {
      websocket: "Connecting",
      camera: null,
      pipeline: null,
      inferenceAt: null,
      heartbeatAt: null,
    };

    if (typeof this.attachShadow === "function") {
      this.attachShadow({ mode: "open" });
      this._render();
    }
  }

  connectedCallback() {
    this._connected = true;
    this._startClient();
  }

  disconnectedCallback() {
    this._connected = false;
    this._releaseClient();
  }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === "ws-url" && oldValue !== newValue && this._connected) {
      this._releaseClient();
      this._health = {
        websocket: "Connecting",
        camera: null,
        pipeline: null,
        inferenceAt: null,
        heartbeatAt: null,
      };
      this._render();
      this._startClient();
    }
  }

  _startClient() {
    if (!this._connected || this._lease) return;
    const url = this.getAttribute("ws-url");
    if (!url) {
      this._health.websocket = "Not configured";
      this._render();
      return;
    }

    this._lease = acquireParkingWebSocketClient(url, {
      onLivenessChange: ({ status, heartbeat, lastHeartbeatAt }) => {
        this._health.websocket = status;
        if (heartbeat) this._applyHealthMessage(heartbeat, lastHeartbeatAt);
        this._render();
      },
      onHeartbeat: (message) => {
        this._applyHealthMessage(message, Date.now());
        this._render();
      },
      onPipelineStatus: (message) => {
        if (typeof (message.pipeline_status ?? message.status) === "string") {
          this._health.pipeline = message.pipeline_status ?? message.status;
        }
        if (typeof message.camera_status === "string") {
          this._health.camera = message.camera_status;
        }
        this._render();
      },
      onParkingStatus: (message) => {
        this.dispatchEvent(new CustomEvent("parking-status", {
          detail: message,
          bubbles: true,
          composed: true,
        }));
      },
    });
  }

  _releaseClient() {
    this._lease?.release();
    this._lease = null;
  }

  _applyHealthMessage(message, receivedAt) {
    if (typeof message.camera_status === "string") {
      this._health.camera = message.camera_status;
    }
    if (typeof message.pipeline_status === "string") {
      this._health.pipeline = message.pipeline_status;
    }
    if (Object.hasOwn(message, "last_inference_at")) {
      this._health.inferenceAt = message.last_inference_at;
    }
    if (receivedAt instanceof Date) receivedAt = receivedAt.getTime();
    if (Number.isFinite(receivedAt)) this._health.heartbeatAt = receivedAt;
  }

  _render() {
    if (!this.shadowRoot) return;
    const health = this._health;
    const entries = [
      ["WebSocket", health.websocket, "websocket"],
      ["Camera", health.camera, "camera"],
      ["YOLO pipeline", health.pipeline, "pipeline"],
      ["Latest inference", this._formatTime(health.inferenceAt), "inference"],
      ["Last heartbeat received", this._formatTime(health.heartbeatAt), "heartbeat"],
    ];
    const items = entries.map(([label, value, key]) => `
      <div class="item">
        <dt>${label}</dt>
        <dd data-key="${key}" data-state="${this._stateClass(value)}">${this._escape(value ?? "Unknown")}</dd>
      </div>`).join("");

    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; color: var(--sp-health-text, #e8edf3); font: inherit; }
        .panel { box-sizing: border-box; border: 1px solid var(--sp-health-border, #334155);
          border-radius: 14px; padding: 18px; background: var(--sp-health-surface, #111827); }
        h2 { margin: 0 0 14px; font-size: 1rem; font-weight: 650; }
        dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
          gap: 10px; margin: 0; }
        .item { min-width: 0; padding: 12px; border-radius: 10px;
          background: var(--sp-health-item, #1f2937); }
        dt { margin-bottom: 7px; color: var(--sp-health-muted, #9ca3af); font-size: .78rem; }
        dd { display: flex; align-items: center; gap: 8px; margin: 0;
          overflow-wrap: anywhere; font-size: .93rem; font-weight: 600; }
        dd[data-key="websocket"]::before, dd[data-key="camera"]::before,
        dd[data-key="pipeline"]::before { content: ""; width: 8px; height: 8px;
          flex: 0 0 8px; border-radius: 50%; background: #64748b; }
        dd[data-state="good"]::before { background: #22c55e; }
        dd[data-state="bad"]::before { background: #ef4444; }
        dd[data-state="stale"]::before, dd[data-state="pending"]::before { background: #eab308; }
        @media (prefers-reduced-motion: no-preference) {
          dd[data-state="stale"]::before { box-shadow: 0 0 0 3px rgb(234 179 8 / 15%); }
        }
      </style>
      <section class="panel" aria-label="System health">
        <h2>System health</h2>
        <dl aria-live="polite">${items}</dl>
      </section>`;
  }

  _formatTime(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return new Date(value).toLocaleString();
    }
    if (typeof value === "string" && value.length > 0) {
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
    }
    return "No data received";
  }

  _stateClass(value) {
    const normalized = String(value ?? "").toLowerCase();
    if (["connected", "running"].includes(normalized)) return "good";
    if (["disconnected", "error", "failed"].includes(normalized)) return "bad";
    if (["stale", "starting", "stopping", "connecting"].includes(normalized)) return "stale";
    return "neutral";
  }

  _escape(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[character]);
  }
}

if (globalThis.customElements && !globalThis.customElements.get("smartparking-health-monitor")) {
  globalThis.customElements.define("smartparking-health-monitor", SmartParkingHealthMonitor);
}
