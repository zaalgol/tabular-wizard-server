from app.repositories.user_repository import UserRepository
from app.services.token_serivce import TokenService
from flask import jsonify, make_response
from datetime import timedelta
from app.models.user import User

class UsersService():
    @staticmethod
    def login(email, password):
        user = UserRepository.get_user_by_email_and_password(email, password)
        if user:
            access_token = TokenService.create_jwt_token(user.user_id)
            
            response = make_response(jsonify({"message": "Login successful", "access_token": access_token}), 200)
            return response

        return jsonify({'message': 'Invalid credentials'}), 401

    @staticmethod
    def create_user(email, password):
        user = UserRepository.create_user(email, password)  # Password should be hashed
        return user
    
    @staticmethod
    def seed_admin_user():
       UserRepository.seed_admin_user()