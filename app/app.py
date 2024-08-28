from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.config.config import Config
from pymongo import MongoClient
# from app.services.init_service import InitService
from fastapi_jwt_auth import AuthJWT
from fastapi_socketio import SocketManager

socketio = SocketManager()

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
    app = FastAPI()
    app.state.config = Config

    # Initialize MongoDB client and set it in app state
    app.state.db = generate_mongo_client()
    
    socketio.attach(app)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.routes.api import router as main_router
    app.include_router(main_router)

    # InitService(app)
    return app
