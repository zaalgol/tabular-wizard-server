from app.services.user_service import UsersService
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from app.config.config import Config
from flask_migrate import Migrate

def generate_flask_server():
    app = Flask(__name__)
    app.config.from_object(Config)

    db = SQLAlchemy(app)
    Migrate(app, db)
    JWTManager(app)

    CORS(app)

    from routes import routes
    
    UsersService.seed_admin_user()
    return app

app = generate_flask_server()
