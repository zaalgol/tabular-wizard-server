class WebsocketService:
    _instance = None

    def __init__(self, app):
        self.app = app

    def __new__(cls, app):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def emit(self, event, data):
         await self.app.state.socketio.emit(event, data)

