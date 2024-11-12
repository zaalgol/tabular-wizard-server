import logging
from fastapi_socketio import SocketManager

logger = logging.getLogger(__name__)

def create_socketio(app):
    return SocketManager(
        app,
        cors_allowed_origins=["*"], # ["http://localhost:5173"],
        async_mode='asgi',
        mount_location='/socket.io',
        transports=['websocket']
        # logger=True,
        # engineio_logger=True
    )
