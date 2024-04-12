from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask import jsonify, make_response
from app.entities.ai_model import AiModel
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
    
    dataset = request.json.get('dataset', None)
    model_name = request.json.get('modelName', None)
    description = request.json.get('description', None)
    target_column = request.json.get('targetColumn', None)
    model_type = request.json.get('modelType', None)
    training_speed = request.json.get('trainingSpeed', None)
    ai_model = AiModel(user_id=user_id, model_name=model_name, description=description, model_type=model_type, training_speed=training_speed, target_column=target_column)

    ai_model_service.train_model(ai_model, dataset)

    return {}, 200, {}

@bp.route('/api/userModels/', methods=['GET'])
@jwt_required()
def get_user_models():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        return {}, 200, {}
    
    user_id =  tokenService.extract_user_id_from_token()
    user = user_service.get_user_by_id(user_id)
    if not user:
        return {}, 401, {}
    models =  ai_model_service.get_user_ai_models_by_id(user_id)
    return make_response(jsonify({"models": models}), 200)


@bp.route('/api/model', methods=['GET'])
@jwt_required()
def get_user_model():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        return {}, 200, {}
    model_name = request.args.get('model_name')
    user_id =  tokenService.extract_user_id_from_token()
    user = user_service.get_user_by_id(user_id)
    if not user:
        return {}, 401, {}
    model =  ai_model_service.get_user_model_by_user_id_and_model_name(user_id, model_name)
    return make_response(jsonify({"model": model}), 200)


@bp.route('/api/inference//', methods=['POST'])
@jwt_required()
def infrernce():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        return {}, 200, {}
    user_id =  tokenService.extract_user_id_from_token()
    user = user_service.get_user_by_id(user_id)
    if not user:
        return {}, 401, {}
    
    dataset = request.json.get('dataset', None)
    model_name = request.json.get('modelName', None)
    file_name = request.json.get('fileName', None)

    ai_model_service.inference(user_id=user_id, model_name=model_name, file_name=file_name, dataset=dataset)

    return {}, 200, {}

@bp.route('/download/<filename>')
def download_file(filename):
    verify_jwt_in_request(locations='query_string')
    # The token has been validated, proceed with sending the file
    user_id = get_jwt_identity()  # If you need to use user information from the token
    model_name = request.args.get('model_name')
    return ai_model_service.downloadInferenceFile(user_id, model_name, filename)

