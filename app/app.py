import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.config.config import Config
from motor.motor_asyncio import AsyncIOMotorClient
from app.socket import create_socketio
from app.routes.api import router as main_router


# Configure logging
logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)

def init_system(app):
    logger.info("Seeding initial data.")
    from app.services.init_service import InitService
    init_service = InitService(app)
    init_service.seed_admin_user()
    init_service.seed_quest_user()

def generate_mongo_client():
    MONGODB_URI = Config.MONGODB_URI

    if int(Config.IS_MONGO_LOCAL):
        mongo_client = AsyncIOMotorClient(MONGODB_URI)
    else:
        mongo_client = AsyncIOMotorClient(
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

    # Seed initial data
    init_system(app)

    # Configure CORS middleware **before** attaching SocketManager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  #  Or specify exact origins like ["http://localhost:5173"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("Attaching SocketManager to the app.")
    # Attach SocketManager to the app
    app.state.socketio = create_socketio(app)

    logger.info("Including main router.")
    app.include_router(main_router)

    app_instance = app  # Store the app instance globally
    return app