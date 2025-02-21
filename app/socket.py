# import logging
# from fastapi_socketio import SocketManager

# logger = logging.getLogger(__name__)

# def create_socketio(app):
#     return SocketManager(
#         app,
#         cors_allowed_origins=["http://localhost:5173", '*'],#'*', # ["http://localhost:5173"],
#         async_mode='asgi',
#         mount_location='/socket.io',
#         transports=['websocket'],
#         logger=True
#         # engineio_logger=True
#     )
# app/socket.py
# app/socket.py
# app/socket.py
import logging
from fastapi_socketio import SocketManager
from app.services.token_service import TokenService

logger = logging.getLogger(__name__)

def create_socketio(app):
    socket_manager = SocketManager(
        app,
        cors_allowed_origins=["http://localhost:5173", 'http://localhost:5174'],
        async_mode='asgi',
        mount_location='/socket.io',
        transports=['websocket'],
        logger=True
    )

    # Access the underlying socketio server
    sio = socket_manager._sio

    @sio.on("connect")
    async def connect(sid, environ, auth):
        """
        Called automatically when a new socket connects.
        We decode the token and join the user’s room.
        """
        # You may need to debug-print 'auth' or 'environ' to see exactly where 
        # your token is. Typically 'auth' is the object you set in 
        # socket.io-client: auth: { token: <some-jwt> }

        logger.info(f"Socket connected: {sid}, auth={auth}")

        token = None
        if auth and "token" in auth:
            token = auth["token"]
        else:
            # Alternatively, you might parse from environ here
            pass

        if not token:
            logger.warning("No token found at connect; skipping room join")
            return

        # Decode token -> user_id
        token_service = TokenService(app.state.db)
        try:
            user_id = await token_service.extract_user_id_from_token(token)
        except Exception as e:
            logger.warning(f"Cannot decode token at connect: {e}")
            return

        # If good user_id, join the user’s room
        logger.info(f"SID={sid} joining room {user_id}")
        await sio.enter_room(sid, user_id)

    @sio.on("disconnect")
    async def disconnect(sid):
        logger.info(f"Socket disconnected: {sid}")

    return socket_manager
