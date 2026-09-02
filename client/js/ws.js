/**
 * WebSocket client with reconnect. Messages are JSON objects with a `type`.
 */
export class WhiteboardSocket {
  constructor(path = "/api/v1/whiteboard/ws") {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this.url = `${proto}//${location.host}${path}`;
    this.ws = null;
    this.onMessage = () => {};
    this.onOpen = () => {};
    this.onClose = () => {};
    this._backoff = 500;
    this._closedByUser = false;
  }

  connect() {
    this._closedByUser = false;
    return new Promise((resolve) => {
      const ws = new WebSocket(this.url);
      this.ws = ws;
      ws.onopen = () => {
        this._backoff = 500;
        this.onOpen();
        resolve();
      };
      ws.onmessage = (ev) => {
        let data;
        try { data = JSON.parse(ev.data); } catch { return; }
        this.onMessage(data);
      };
      ws.onclose = () => {
        this.onClose();
        if (this._closedByUser) return;
        setTimeout(() => this.connect(), this._backoff);
        this._backoff = Math.min(8000, this._backoff * 2);
      };
      ws.onerror = () => ws.close();
    });
  }

  get isOpen() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  send(obj) {
    if (!this.isOpen) return false;
    this.ws.send(JSON.stringify(obj));
    return true;
  }

  close() {
    this._closedByUser = true;
    this.ws?.close();
  }
}
