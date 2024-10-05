# app/services/token_service.py

from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from starlette.status import HTTP_401_UNAUTHORIZED

# Replace with your actual secret key. Ensure this is kept secure!
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize OAuth2PasswordBearer with auto_error=False to handle missing tokens gracefully
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class TokenService:
    @staticmethod
    def create_jwt_token(user_id: str, expires_delta: timedelta = None):
        """
        Create a JWT token for a given user ID with an optional expiration delta.
        """
        to_encode = {"sub": user_id}
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            # Default to a long expiration time; adjust as needed
            expire = datetime.utcnow() + timedelta(minutes=1500)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str):
        """
        Decode a JWT token and extract the user ID.
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user_id
        except JWTError:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    async def extract_user_id_from_token(request: Request, token: str = Depends(oauth2_scheme)):
        """
        Extract the user ID from the JWT token. The token can be provided either in the
        Authorization header or as a query parameter named 'Authorization'.
        """
        if not token:
            # Attempt to retrieve the token from query parameters
            token = request.query_params.get('Authorization')

        if not token:
            # If token is still not found, raise an unauthorized error
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Decode and return the user ID from the token
        return TokenService.decode_token(token)
