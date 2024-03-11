from app.services.user_service import UsersService
from flask import request, jsonify, make_response
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from app import app

@app.route('/', methods=['GET'])
@jwt_required()
def hello_world():
    return 'Hello, World!'


@app.route('/api/login/', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        return {}, 200, {}

    email = request.json.get('email', None)
    password = request.json.get('password', None)
    if not email or not password:
        return jsonify({'message': 'Invalid credentials'}), 401
    return UsersService.login(email, password)

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    user = next((user for user in users if user['id'] == current_user_id), None)

    if user:
        return jsonify({'message': f'Hello, {user["email"]}!'}), 200

    return jsonify({'message': 'User not found'}), 404
