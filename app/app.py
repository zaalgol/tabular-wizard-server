import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.config.config import Config
from pymongo import MongoClient
from app.socket import create_socketio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mongo_client():
    MONGODB_URI = Config.MONGODB_URI

    if int(Config.IS_MONGO_LOCAL):
        mongo_client = MongoClient(MONGODB_URI)
    else:
        mongo_client = MongoClient(
            MONGODB_URI,
            tls=True,
            retryWrites=False,
            tlsCAFile="global-bundle.pem",  # Adjust path accordingly
            socketTimeoutMS=60000,
            connectTimeoutMS=60000
        )
    db = mongo_client['tabular-wizard-db']
    return db

def create_app():
    global app_instance
    app = FastAPI()
    app.state.config = Config

    logger.info("Initializing MongoDB client.")
    # Initialize MongoDB client and set it in app state
    app.state.db = generate_mongo_client()

    # Configure CORS middleware **before** attaching SocketManager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Or specify exact origins like ["http://localhost:5173"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("Attaching SocketManager to the app.")
    # Attach SocketManager to the app
    app.state.socketio = create_socketio(app)

    logger.info("Including main router.")
    from app.routes.api import router as main_router
    app.include_router(main_router)

    app_instance = app  # Store the app instance globally
    return app

def get_app():
    global app_instance
    if app_instance is None:
        raise RuntimeError("App has not been created yet")
    return app_instance
