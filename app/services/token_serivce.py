import jwt
import datetime

# Your "secret" key used for encoding and decoding the JWT. Keep this safe.
SECRET_KEY = "your_secret_key_here"

def create_jwt_token(user_id):
    # Payload data you want to encode within the JWT
    payload = {
        "user_id": user_id,  # Example user identifier
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1),  # Expiration time (1 day from now)
        "iat": datetime.datetime.utcnow()  # Issued at time
    }

    # Encode the payload with your secret key
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    return token
