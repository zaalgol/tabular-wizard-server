from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.config import Config
from app.socket import create_socketio
from app.routes.api import router as main_router
from app.services.init_service import InitService

import logging
logger = logging.getLogger(__name__)

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
    return mongo_client['tabular-wizard-db']

async def lifespan(app: FastAPI):
    logger.info("Initializing MongoDB client.")
    app.state.db = generate_mongo_client()

    # Seed initial data
    logger.info("Seeding initial data.")
    init_service = InitService(app)
    await init_service.seed_admin_user()
    await init_service.seed_quest_user()

    yield  # This allows the app to start

    logger.info("Cleaning up resources...")

async def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)  # Attach lifespan here
    app.state.config = Config

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Customize as needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("Attaching SocketManager to the app.")
    app.state.socketio = create_socketio(app)

    logger.info("Including main router.")
    app.include_router(main_router)

    return app
