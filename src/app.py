from flask import Flask
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_migrate import Migrate
from pymongo import MongoClient
from app.ai.services.model_service import ModelService
from app.ai.config.config import Config 
from flask_socketio import SocketIO

jwt = JWTManager()
migrate = Migrate()
socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    mongo_client = MongoClient(app.config['MONGODB_URI'])
    db = mongo_client.get_default_database()
    app.db = db
    
    jwt.init_app(app)
    socketio.init_app(app)  # Attach SocketIO to Flask app
    CORS(app, supports_credentials=True)

    from app.ai.routes.api import bp as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
