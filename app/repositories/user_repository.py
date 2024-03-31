from bson import ObjectId
from flask import current_app
from datetime import datetime, UTC

class UserRepository:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def db(self):
        return current_app.db

    @property
    def users_collection(self):
        return self.db['users']
    
    def get_user_by_id(self, user_id):
        return self.users_collection.find_one({"_id": ObjectId(user_id), "isDeleted": {"$ne": True}})

    def get_user_by_username(self, username):
        return self.users_collection.find_one({"username": username, "isDeleted": {"$ne": True}})

    def get_user_by_email(self, email):
        return self.users_collection.find_one({"email": email, "isDeleted": {"$ne": True}})
    
    def get_user_by_email_and_password(self, email, password):
        return self.users_collection.find_one({"email": email, "password": password, "isDeleted": {"$ne": True}})

    def create_user(self, email, password):
        user_exists = self.users_collection.find_one({"email": email})
        if user_exists:
            return f"user with email {email} already exist"
        user = {
            "email": email,
            "password": password,  # Ensure password is hashed appropriately
        }
        result = self.users_collection.insert_one(user)
        return {"_id": result.inserted_id, **user}
    
