from flask_jwt_extended import create_access_token
from flask import jsonify, make_response

class TokenService():
    @staticmethod
    def create_jwt_token(user_id):
        access_token = create_access_token(identity=user_id)
        return access_token
