import logging
from fastapi_socketio import SocketManager

logger = logging.getLogger(__name__)

def create_socketio(app):
    logger.info("Creating SocketManager.")
    return SocketManager(app)