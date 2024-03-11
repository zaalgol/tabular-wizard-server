from flask import Flask, request, jsonify, json, make_response, jsonify
from werkzeug.exceptions import HTTPException
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from app import app

# app = Flask(__name__)
# # CORS(app) # This will enable CORS for all routes
# CORS(app, supports_credentials=True)
# # from app import app

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'


app.config['JWT_SECRET_KEY'] = 'your-secret-key'
jwt = JWTManager(app)

# Mock user data
users = [
    {'id': 1, 'username': 'user1', 'password': 'password1'},
    {'id': 2, 'username': 'user2', 'password': 'password2'},
]


@app.route('/api/login/', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        return {}, 200, {}

    username = request.json.get('username', None)
    password = request.json.get('password', None)

    user = next((user for user in users if user['username'] == username and user['password'] == password), None)
    # Simple authentication logic (replace with actual validation against user data)
    if user:
        access_token = create_access_token(identity=user['id'])
        
        response = make_response(jsonify({"message": "Login successful", "access_token": access_token}), 200)
        # response.set_cookie('access_token', access_token, httponly=True)
        return response

    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    user = next((user for user in users if user['id'] == current_user_id), None)

    if user:
        return jsonify({'message': f'Hello, {user["username"]}!'}), 200

    return jsonify({'message': 'User not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)