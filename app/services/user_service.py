from app.repositories.user_repository import UserRepository
from app.services.hashing_service import PasswordHasher
from app.services.token_service import TokenService
from fastapi.responses import JSONResponse

class UserService:
    _instance = None

    def __new__(cls, db):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, db):
        # Ensure __init__ is only called once
        if self.__initialized:
            return
        self.__initialized = True
        self.user_repository = UserRepository(db)
        self.token_service = TokenService()

    def login(self, email, password):
        user = self.user_repository.get_user_by_email(email)
        if user:
            is_valid_password = PasswordHasher.check_password(user['password'], password)
            if is_valid_password:
                access_token = self.token_service.create_jwt_token(str(user['_id']))
                return JSONResponse({"message": "Login successful", "access_token": access_token}, status_code=200)
        return JSONResponse({'message': 'Invalid credentials'}, status_code=401)
    
    def create_user(self, email, password):
        hashed_password = PasswordHasher.hash_password(password)
        user = self.user_repository.create_user(email, hashed_password) 
        return user
    
    def get_user_by_id(self, user_id):
        return self.user_repository.get_user_by_id(user_id)
