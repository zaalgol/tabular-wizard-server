from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS

def generate_flask_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)
    return app

app = generate_flask_app()
from app import routes