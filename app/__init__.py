from flask import Flask
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_migrate import Migrate
from pymongo import MongoClient
from app.config.config import Config 

# db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.logger.debug
    app.config.from_object(Config)
    mongo_client = MongoClient(app.config['MONGODB_URI'])
    db = mongo_client.get_default_database()
    app.db = db
    
    jwt.init_app(app)
    # migrate.init_app(app, db)
    
    # Apply CORS to the entire app with support for credentials
    CORS(app, supports_credentials=True)
    # CORS(app, resources={r"/*": {"origins": "*"}}) # also not solving the CORS issue

    from app.routes.api import bp as main_blueprint
    app.register_blueprint(main_blueprint)
    return app
