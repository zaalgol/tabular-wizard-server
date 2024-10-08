from bson import ObjectId
from datetime import datetime, timezone
from pymongo.database import Database

class UserRepository:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db: Database):
        self._db = db

    @property
    def db(self):
        return self._db

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
        try:
            user_exists = self.users_collection.find_one({"email": email})
            if user_exists:
                return f"user with email {email} already exist"
            user = {
                "email": email,
                "password": password,  # Ensure password is hashed appropriately
            }
            result = self.users_collection.insert_one(user)
            return {"_id": result.inserted_id, **user}
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
