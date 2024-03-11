from flask import Flask, request, jsonify, make_response
from datetime import timedelta
from app import app


class UsersService():
    def login(username, password):
        if not username or not password:
            return jsonify({"message": "Unauthorized"}), 401
        # Simple authentication logic (replace with actual validation against user data)
        
        if data['username'] == 'admin' and data['password'] == 'admin':
            # Generate a token. This is just a placeholder. Use a secure method for real applications.
            token = "secureRandomTokenHere"

            # Create response object
            response = make_response(jsonify({"message": "Login successful"}), 200)

            # Set token in httpOnly cookie
            response.set_cookie('auth_token', token, httponly=True, max_age=timedelta(days=1))

            return response
        else:
            return jsonify({"message": "Unauthorized"}), 401