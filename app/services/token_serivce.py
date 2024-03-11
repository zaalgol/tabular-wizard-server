from flask_jwt_extended import create_access_token
from flask import jsonify, make_response

class TokenService():
    @staticmethod
    def create_jwt_token(user_id):
        access_token = create_access_token(identity=user_id)
        
        response = make_response(jsonify({"message": "Login successful", "access_token": access_token}), 200)
        return response
#         # Payload data you want to encode within the JWT
#         payload = {
#             "user_id": user_id,  # Example user identifier
#             "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=3),  # Expiration time (1 day from now)
#             "iat": datetime.datetime.utcnow()  # Issued at time
#         }
# 
#         # Encode the payload with your secret key
#         token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
# 
#         return token
