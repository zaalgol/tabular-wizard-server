# app/services/websocket_service.py
from app.services.websocket_manager import WebSocketManager

class WebsocketService:
    _instance = None

    def __new__(cls, websocket_manager: WebSocketManager):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.websocket_manager = websocket_manager
        return cls._instance

    async def emit(self, event, data):
        await self.websocket_manager.broadcast(event, data)
