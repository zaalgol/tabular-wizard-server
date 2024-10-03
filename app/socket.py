# app/socket.py

import logging
from fastapi_socketio import SocketManager

logger = logging.getLogger(__name__)

# def create_socketio(app):
#     logger.info("Creating SocketManager.")
#     return SocketManager(
#         app,
#         cors_allowed_origins=None,  # Allows all origins
#         transports=['websocket'],
#         logger=True,                # Enable logging
#         engineio_logger=True        # Enable engine.io logging
#     )


def create_socketio(app):
    return SocketManager(
        app,
        cors_allowed_origins=["http://localhost:5173"],
        async_mode='asgi',
        mount_location='/socket.io',
        transports=['websocket']
        # logger=True,
        # engineio_logger=True
    )
