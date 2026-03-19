"""WebSocket hub and /ws endpoint."""

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


class WebSocketHub:
    """Manages connected WebSocket clients and broadcasts events."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        logger.info("WS client connected (%d total)", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("WS client disconnected (%d total)", len(self._clients))

    async def broadcast(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, default=str)
        dead: set[WebSocket] = set()
        for client in list(self._clients):
            try:
                await client.send_text(payload)
            except Exception:
                dead.add(client)
        self._clients -= dead


ws_hub = WebSocketHub()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_hub.disconnect(websocket)
    except Exception:
        ws_hub.disconnect(websocket)
