from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from app.services.user_service import UsersService
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_cors import CORS
from flask_cors import cross_origin

# Create a Blueprint
bp = Blueprint('main', __name__)
CORS(bp)

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
    usersService = UsersService()
    return usersService.login(email, password)
