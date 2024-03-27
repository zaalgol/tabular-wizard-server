from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from app.services.user_service import UsersService
from flask_cors import CORS

import pandas as pd

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

@bp.route('/api/trainModel/', methods=['POST'])
def train_model():
    dataset = request.json.get('dataset', None)
    dataset_json = request.json.get('dataset', None)
    
    if dataset_json is None:
        return {"error": "No dataset provided"}, 400

    # Convert the dataset to a DataFrame
    # Assuming the first row of the dataset is the header
    headers = dataset_json[0]
    data_rows = dataset_json[1:]
    df = pd.DataFrame(data_rows, columns=headers)
    df.to_csv("temp.csv")

    target_column = request.json.get('targetColumn', None)
    model_type = request.json.get('modelType', None)
    training_speed = request.json.get('trainingSpeed', None)
    return {}, 200, {}

