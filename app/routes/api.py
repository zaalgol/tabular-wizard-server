from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from app.services.user_service import UsersService
from flask_cors import CORS

# Create a Blueprint
bp = Blueprint('main', __name__)
CORS(bp)

# Instantiate UsersService singleton
users_service = UsersService()

@bp.route('/', methods=['GET'])
@jwt_required()
def hello_world():
    return 'Hello, World!'

@bp.route('/api/login/', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        return {}, 200, {}

    email = request.json.get('email', None)
    password = request.json.get('password', None)
    if not email or not password:
        return jsonify({'message': 'Invalid credentials'}), 401

    return users_service.login(email, password)

