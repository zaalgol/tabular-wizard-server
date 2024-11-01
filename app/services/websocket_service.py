
from app.app import get_app
class WebsocketService:
    _instance = None

    def __init__(self):
        self.socketio = get_app().state.socketio 

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def emit(self, event, data):
         await self.socketio.emit(event, data)


