from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_migrate import Migrate
from app.config.config import Config  # Adjust import path as necessary

db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)
    
    # Apply CORS to the entire app with support for credentials
    CORS(app, supports_credentials=True)
    # CORS(app, resources={r"/*": {"origins": "*"}}) # also not solving the CORS issue

    from app.routes.api import bp as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
