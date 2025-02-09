# class WebsocketService:
#     _instance = None

#     def __init__(self, app):
#         self.app = app

#     def __new__(cls, app):
#         if not cls._instance:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     async def emit(self, event, data):
#          await self.app.state.socketio.emit(event, data)

# app/services/websocket_service.py

# app/services/websocket_service.py

class WebsocketService:
    _instance = None

    def __init__(self, app):
        self.app = app

    def __new__(cls, app):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def emit(self, event: str, data: dict, user_id: str = None):
        """
        If user_id is provided, emit only to that user’s room.
        Otherwise, broadcast to everyone.
        """
        if user_id:
            await self.app.state.socketio.emit(event, data, room=user_id)
        else:
            await self.app.state.socketio.emit(event, data)



