from flask import Flask
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_migrate import Migrate
from pymongo import MongoClient
from app.config.config import Config 
from flask_socketio import SocketIO
import logging
from logging.handlers import RotatingFileHandler
from app.services.init_service import InitService

jwt = JWTManager()
migrate = Migrate()
socketio = SocketIO(cors_allowed_origins="*")

def generate_mongo_client(app):
    MONGODB_URI = app.config['MONGODB_URI']
    
    if int(app.config['IS_MONGO_LOCAL']):
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
    app.db = db
    
def set_logger(app):
    
    # Set up the logging handler
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)

    # Create a custom logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Attach the handler to the application's logger
    app.logger.addHandler(handler)

    # Example log messages
    app.logger.info('Logging is configured!')
    app.logger.error('Sample error message')
    
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    generate_mongo_client(app)
    # mongo_client = MongoClient(app.config['MONGODB_URI'])
    # db = mongo_client.get_default_database()
    # app.db = db
    
    jwt.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*", async_mode='gevent')  # Attach SocketIO to Flask app
    # socketio.init_app(app, cors_allowed_origins="*",  engineio_logger=True, logger=True)  # Attach SocketIO to Flask app
    # socketio.init_app(app, cors_allowed_origins="*") 
    CORS(app, supports_credentials=True, origins="*")

    from app.routes.api import bp as main_blueprint
    app.register_blueprint(main_blueprint)

    InitService(app)
    return app
