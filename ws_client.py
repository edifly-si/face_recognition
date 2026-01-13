import websocket
import json
import threading
import time

class WSClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.connected = False
        self._connect()

    def _connect(self):
        def run():
            while True:
                try:
                    self.ws = websocket.WebSocket()
                    self.ws.connect(self.url, timeout=3)
                    self.connected = True
                    print("[WS] Connected")
                    break
                except Exception as e:
                    print("[WS] Retry...", e)
                    time.sleep(2)

        threading.Thread(target=run, daemon=True).start()

    def send(self, data):
        if not self.connected:
            return
        try:
            self.ws.send(json.dumps(data))
        except Exception as e:
            print("[WS] Send failed:", e)
            self.connected = False
            self._connect()
