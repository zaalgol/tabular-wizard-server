from datetime import timedelta
from fastapi.responses import JSONResponse
from app.logger_setup import setup_logger
from app.repositories.user_repository import UserRepository
from app.services.hashing_service import PasswordHasher
from app.services.token_service import TokenService
from app.config.config import Config 

logger = setup_logger(__name__)

class UserService:
    _instance = None

    def __new__(cls, db):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, db):
        self.db = db
        # Ensure __init__ is only called once
        if self.__initialized:
            return
        self.__initialized = True
        self.db = db
        self.user_repository = UserRepository(db)
        self.token_service = TokenService(db)

    async def login(self, email, password):
        user = await self._authenticate_user(email, password)
        if not user:
            return JSONResponse(
                {'message': 'Invalid credentials'}, 
                status_code=401
            )
        
        return await self._create_login_response(user)

    async def _authenticate_user(self, email, password):
        user = await self.user_repository.get_user_by_email(email)
        if user and PasswordHasher.check_password(user['password'], password):
            return user
        return None

    async def _create_login_response(self, user):
        tokens = await self._generate_tokens(str(user['_id']))
        
        response = JSONResponse({
            "message": "Login successful",
            "access_token": tokens['access_token']
        }, status_code=200)
        
        self._set_refresh_token_cookie(response, tokens['refresh_token'], tokens['refresh_token_expires'])
        return response

    async def _generate_tokens(self, user_id):
        access_token_expires = timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
        refresh_token_expires = timedelta(days=Config.REFRESH_TOKEN_EXPIRE_DAYS)
        
        access_token = await self.token_service.create_access_token(
            user_id, 
            expires_delta=access_token_expires
        )
        refresh_token = await self.token_service.create_refresh_token(
            user_id, 
            expires_delta=refresh_token_expires
        )
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'refresh_token_expires': refresh_token_expires
        }

    def _set_refresh_token_cookie(self, response, refresh_token, refresh_token_expires):
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite='lax',
            max_age=int(refresh_token_expires.total_seconds())
        )

    async def create_user(self, email, password):
        hashed_password = PasswordHasher.hash_password(password)
        user = await self.user_repository.create_user(email, hashed_password)
        if user is None:
            return None
        return user
    
    async def get_user_by_id(self, user_id):
        return await self.user_repository.get_user_by_id(user_id)

    def validate_user_password(self, user, password):
        return PasswordHasher.check_password(user['password'], password)

    async def update_user_password(self, user_id, new_password):
        hashed_password = PasswordHasher.hash_password(new_password)
        return await self.user_repository.update_password(user_id, hashed_password)
    
    async def request_password_reset(self, email: str):
        """
        Generate a password-reset token for the user, 
        and return it (or send email).
        """
        user = await self.user_repository.get_user_by_email(email)
        if not user:
            # In real world, you might not want to reveal 
            # that the email doesn't exist
            return None

        # Generate a "reset" token
        reset_token = await self.token_service.create_reset_password_token(str(user['_id']))

        # In real code, you would integrate with an email sending service here:
        #   send_password_reset_email(to=email, token=reset_token)
        #
        # For now, just return the token so the client can show or handle it
        return reset_token

    async def reset_password_with_token(self, reset_token: str, new_password: str):
        """
        Validate the reset token, find the user, update password.
        """
        # Validate (decode) token and ensure it is not used/expired
        user_id = await self.token_service.validate_and_delete_reset_token(reset_token)

        hashed_password = PasswordHasher.hash_password(new_password)
        updated_user = await self.user_repository.update_password(user_id, hashed_password)
        return updated_user
