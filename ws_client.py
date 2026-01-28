import websocket
import json
import threading
import time

class WSClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.connected = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            if not self.connected:
                try:
                    print("[WS] Connecting...")
                    self.ws = websocket.WebSocket()
                    self.ws.settimeout(5)
                    self.ws.connect(self.url)
                    self.connected = True
                    print("[WS] Connected")
                except Exception as e:
                    print("[WS] Connect failed:", e)
                    self.connected = False
                    time.sleep(3)
                    continue

            # connection watchdog
            try:
                self.ws.ping()
                time.sleep(5)
            except Exception as e:
                print("[WS] Lost connection:", e)
                self.connected = False
                try:
                    self.ws.close()
                except:
                    pass
                time.sleep(2)

    def send(self, data):
        if not self.connected:
            return
        try:
            with self.lock:
                self.ws.send(json.dumps(data))
        except Exception as e:
            print("[WS] Send error:", e)
            self.connected = False
