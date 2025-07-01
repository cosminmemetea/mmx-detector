import asyncio
from typing import Any, Dict, List

from fastapi import WebSocket


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[WebSocket]] = {}

    async def subscribe(self, task_id: str, websocket: WebSocket) -> None:
        self._subs.setdefault(task_id, []).append(websocket)

    async def unsubscribe(self, task_id: str, websocket: WebSocket) -> None:
        if task_id in self._subs and websocket in self._subs[task_id]:
            self._subs[task_id].remove(websocket)

    async def publish(self, task_id: str, message: Any) -> None:
        for ws in list(self._subs.get(task_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                # drop broken connections
                await self.unsubscribe(task_id, ws)
