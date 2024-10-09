from datetime import datetime, timezone
from pymongo.database import Database

class TokenRepository:
    def __init__(self, db: Database):
        self.collection = db['refresh_tokens']

    def save_refresh_token(self, token_id, user_id, expires_at):
        self.collection.insert_one({
            "_id": token_id,
            "user_id": user_id,
            "expires_at": expires_at
        })

    def delete_refresh_token(self, token_id):
        self.collection.delete_one({"_id": token_id})

    def get_refresh_token(self, token_id):
        return self.collection.find_one({"_id": token_id})