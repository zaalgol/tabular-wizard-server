from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask import jsonify, make_response
from app.entities.model import Model
from app.services.model_service import ModelService
from app.services.token_serivce import TokenService
from app.services.user_service import UserService
from flask_cors import CORS

from flask_jwt_extended import create_access_token, verify_jwt_in_request
# Create a Blueprint
bp = Blueprint('main', __name__)
CORS(bp)

# Instantiate UsersService singleton
user_service = UserService()
model_service = ModelService()
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
    # training_speed = request.json.get('trainingSpeed', None)
    # ensemble = request.json.get('ensemble', None)
    training_strategy = request.json.get('trainingStrategy', None)
    sampling_strategy = request.json.get('samplingStrategy', None)
    metric = request.json.get('metric', None)
    model = Model(user_id=user_id, model_name=model_name, description=description,
                   model_type=model_type, training_strategy=training_strategy, sampling_strategy=sampling_strategy, target_column=target_column, metric=metric)

    return model_service.train_model(model, dataset)

    

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
    models =  model_service.get_user_models_by_id(user_id)
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
    model =  model_service.get_user_model_by_user_id_and_model_name(user_id, model_name)
    return make_response(jsonify({"model": model}), 200)

@bp.route('/api/model', methods=['OPTIONS', 'DELETE'])
@jwt_required()
def delete_model():
    if request.method == 'OPTIONS':
        # Handle the preflight request
        return {}, 200, {}
    model_name = request.args.get('model_name')
    user_id =  tokenService.extract_user_id_from_token()
    user = user_service.get_user_by_id(user_id)
    if not user:
        return {}, 401, {}
    result =  model_service.delete_model_for_user(user_id, model_name)
    return {}, 200, {}


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

    model_service.inference(user_id=user_id, model_name=model_name, file_name=file_name, dataset=dataset)

    return {}, 200, {}

@bp.route('/download/<filename>')
def download_file(filename):
    verify_jwt_in_request(locations='query_string')
    # The token has been validated, proceed with sending the file
    user_id = get_jwt_identity()  # If you need to use user information from the token
    model_name = request.args.get('model_name')
    file_type = request.args.get('file_type')
    return model_service.downloadFile(user_id, model_name, filename, file_type)

