from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from app.services.ai_model_service import AiModelService
from app.services.token_serivce import TokenService
from app.services.user_service import UserService
from flask_cors import CORS

from flask_jwt_extended import create_access_token, verify_jwt_in_request
# Create a Blueprint
bp = Blueprint('main', __name__)
CORS(bp)

# Instantiate UsersService singleton
user_service = UserService()
ai_model_service = AiModelService()
tokenService = TokenService()

@bp.route('/', methods=['GET'])
# @jwt_required()
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

    return user_service.login(email, password)

@bp.route('/api/trainModel/', methods=['POST'])
@jwt_required()
def train_model():
    user_id =  tokenService.extract_user_id_from_token()
    user = user_service.get_user_by_id(user_id)
    if not user:
        return {}, 401, {}
    model_name = request.json.get('modelName', None)
    dataset = request.json.get('dataset', None)
    
    if dataset is None:
        return {"error": "No dataset provided"}, 400
    
    # Assuming the first row of the dataset is the header
    headers = dataset[0]
    data_rows = dataset[1:200]

    target_column = request.json.get('targetColumn', None)
    model_type = request.json.get('modelType', None)
    training_speed = request.json.get('trainingSpeed', None)
    ai_model_service.train_model(user_id, model_name, headers, data_rows, target_column, model_type, training_speed)

    return {}, 200, {}


