from flask_jwt_extended import create_access_token, verify_jwt_in_request
from datetime import timedelta

class TokenService():
    @staticmethod
    def create_jwt_token(user_id):
        access_token = create_access_token(identity=user_id, expires_delta=timedelta(days=1))
        return access_token
    
    @staticmethod
    def extract_user_id_from_token():
        return verify_jwt_in_request()[1]['sub']
