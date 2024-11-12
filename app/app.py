# app/app.py
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.config.config import Config
from pymongo import MongoClient
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
        mongo_client = MongoClient(MONGODB_URI)
    else:
        mongo_client = MongoClient(
            MONGODB_URI,
            tls=True,
            retryWrites=False,
            tlsCAFile="global-bundle.pem",
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
    app.state.db = generate_mongo_client()

    init_system(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("Including main router.")
    app.include_router(main_router)

    app_instance = app
    return app

def get_app():
    global app_instance
    if app_instance is None:
        raise RuntimeError("App has not been created yet")
    return app_instance
