# app/services/token_service.py

import uuid
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from starlette.status import HTTP_401_UNAUTHORIZED
from app.config.config import Config
from app.repositories.token_repository import TokenRepository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class TokenService:
    def __init__(self, db):
        self.db = db
        self.token_repository = TokenRepository(db)

    async def create_access_token(self, user_id: str, expires_delta: timedelta = None):
        to_encode = {"sub": user_id, "type": "access"}
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, Config.ACCESS_TOKEN_SECRET_KEY, algorithm=Config.ALGORITHM)
        return encoded_jwt

    async def create_refresh_token(self, user_id: str, expires_delta: timedelta = None):
        token_id = str(uuid.uuid4())
        to_encode = {"sub": user_id, "type": "refresh", "jti": token_id}
        expire = datetime.utcnow() + (expires_delta or timedelta(days=Config.REFRESH_TOKEN_EXPIRE_DAYS))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, Config.REFRESH_TOKEN_SECRET_KEY, algorithm=Config.ALGORITHM)

        # Save the token ID and expiration in the database
        await self.token_repository.save_refresh_token(token_id, user_id, expire)

        return encoded_jwt

    def decode_token(self, token: str, expected_type: str):
        try:
            if expected_type == "access":
                secret_key = Config.ACCESS_TOKEN_SECRET_KEY
            elif expected_type == "refresh":
                secret_key = Config.REFRESH_TOKEN_SECRET_KEY
            else:
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid token type")

            payload = jwt.decode(token, secret_key, algorithms=[Config.ALGORITHM])
            user_id: str = payload.get("sub")
            token_type: str = payload.get("type")
            if user_id is None or token_type != expected_type:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return payload
        except:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def refresh_access_token(self, refresh_token: str):
        try:
            payload = self.decode_token(refresh_token, expected_type="refresh")
            token_id = payload.get("jti")
            user_id = payload.get("sub")
            if not token_id or not user_id:
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

            # Verify the token ID exists in the database
            token_record = self.token_repository.get_refresh_token(token_id)
            if not token_record or token_record['user_id'] != user_id:
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

            # Implement token rotation: delete old refresh token
            self.token_repository.delete_refresh_token(token_id)

            # Create new refresh token
            new_refresh_token = await self.create_refresh_token(user_id)

            # Create new access token
            access_token = self.create_access_token(user_id)

            return access_token, new_refresh_token

        except HTTPException as e:
            raise e

    async def extract_user_id_from_token(self, token: str, expected_type: str = "access"):
        if not token:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload = self.decode_token(token, expected_type=expected_type)
        return payload.get("sub")

    async def create_reset_password_token(self, user_id: str, expires_delta: timedelta = None):
        """
        Creates a short-lived JWT for resetting password, 
        then persists the token ID in the DB.
        """
        token_id = str(uuid.uuid4())
        to_encode = {"sub": user_id, "type": "reset", "jti": token_id}

        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))  # Example: 30 mins
        to_encode.update({"exp": expire})

        # Encode using a special key or reuse one (prefer a dedicated key if you want).
        encoded_jwt = jwt.encode(
            to_encode, 
            Config.ACCESS_TOKEN_SECRET_KEY,  # or a separate key e.g. Config.RESET_TOKEN_SECRET_KEY
            algorithm=Config.ALGORITHM
        )

        # Save the reset token in DB
        await self.token_repository.save_reset_token(token_id, user_id, expire)

        return encoded_jwt

    def decode_reset_token(self, token: str):
        """
        Decodes a 'reset' token and returns the payload.
        """
        try:
            payload = jwt.decode(
                token, 
                Config.ACCESS_TOKEN_SECRET_KEY,  # or dedicated reset key
                algorithms=[Config.ALGORITHM]
            )
            user_id: str = payload.get("sub")
            token_type: str = payload.get("type")
            token_id: str = payload.get("jti")

            if token_type != "reset" or user_id is None or token_id is None:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Could not validate reset token",
                )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Could not validate reset token",
            )

    async def validate_and_delete_reset_token(self, token: str):
        """
        Validates the token by decoding it and checking 
        if the token ID is stored in DB. Then removes it (single-use).
        Returns the user_id if valid.
        """
        payload = self.decode_reset_token(token)
        token_id = payload.get("jti")
        user_id = payload.get("sub")

        # Check in DB
        token_record = self.token_repository.get_reset_token(token_id)
        if not token_record or token_record["user_id"] != user_id:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token",
            )
        
        # If everything is good, we delete the token to prevent reuse
        self.token_repository.delete_reset_token(token_id)

        return user_id