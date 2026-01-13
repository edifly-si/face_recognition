import asyncio
import websockets

CLIENTS = set()

async def handler(ws):
    CLIENTS.add(ws)
    print("[WS] Client connected:", ws.remote_address)

    try:
        async for message in ws:
            # broadcast ke semua client (FE)
            for client in CLIENTS:
                if client != ws:
                    await client.send(message)

    except websockets.ConnectionClosed:
        pass
    finally:
        CLIENTS.remove(ws)
        print("[WS] Client disconnected")

async def main():
    print("[WS] Server running on ws://0.0.0.0:3001/ws/face-stream")
    async with websockets.serve(handler, "0.0.0.0", 3001, ping_interval=None):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
