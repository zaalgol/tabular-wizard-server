from fastapi_jwt_auth import AuthJWT
from datetime import timedelta

class TokenService:
    @staticmethod
    def create_jwt_token(user_id, Authorize: AuthJWT):
        access_token = Authorize.create_access_token(subject=user_id, expires_time=timedelta(days=1))
        return access_token
    
    @staticmethod
    def extract_user_id_from_token(Authorize: AuthJWT):
        Authorize.jwt_required()
        return Authorize.get_jwt_subject()
