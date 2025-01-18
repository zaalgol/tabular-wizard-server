from motor.motor_asyncio import AsyncIOMotorDatabase

from app.logger_setup import setup_logger

logger = setup_logger(__name__)

class TokenRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db['refresh_tokens']

    async def save_refresh_token(self, token_id: str, user_id: str, expires_at):
        """Save a refresh token with its ID, associated user ID, and expiration date."""
        try:
            await self.collection.insert_one({
                "_id": token_id,
                "user_id": user_id,
                "expires_at": expires_at
            })
        except Exception as e:
            logger.error(f"Error saving refresh token {token_id}: {e}")

    async def delete_refresh_token(self, token_id: str):
        """Delete a refresh token by its ID."""
        try:
            await self.collection.delete_one({"_id": token_id})
        except Exception as e:
            logger.error(f"Error deleting refresh token {token_id}: {e}")

    async def get_refresh_token(self, token_id: str):
        """Retrieve a refresh token by its ID."""
        try:
            token = await self.collection.find_one({"_id": token_id})
            return token
        except Exception as e:
            logger.error(f"Error retrieving refresh token {token_id}: {e}")
            return None