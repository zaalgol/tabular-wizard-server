from bson import ObjectId
from app.logger_setup import setup_logger
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = setup_logger(__name__)

class UserRepository:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db: AsyncIOMotorDatabase):
        if not hasattr(self, '_db'):
            self._db = db

    @property
    def users_collection(self):
        """Access the 'users' collection from the database."""
        return self._db['users']

    async def get_user_by_id(self, user_id: str):
        """Fetch a user by their ID if they are not marked as deleted."""
        try:
            user = await self.users_collection.find_one({
                "_id": ObjectId(user_id),
                "isDeleted": {"$ne": True}
            })
            return user
        except Exception as e:
            logger.error(f"Error fetching user by id {user_id}: {e}")
            return None

    async def get_user_by_email(self, email: str):
        """Fetch a user by their email if they are not marked as deleted."""
        try:
            user = await self.users_collection.find_one({
                "email": email,
                "isDeleted": {"$ne": True}
            })
            return user
        except Exception as e:
            logger.error(f"Error fetching user by email {email}: {e}")
            return None

    async def create_user(self, email: str, password: str):
        """Create a new user with the provided email and password."""
        try:
            user_exists = await self.users_collection.find_one({"email": email})
            if user_exists:
                logger.info(f"User with email {email} already exists.")
                return None

            user = {
                "email": email,
                "password": password,
                "isDeleted": False  # Default value for new users
            }
            result = await self.users_collection.insert_one(user)
            logger.info(f"User created with id {result.inserted_id}")
            return {"_id": str(result.inserted_id), **user}
        except Exception as e:
            logger.error(f"Exception creating user {email}: {e}")
            return None

    async def update_password(self, user_id: str, new_password: str):
        """Update the password of a user by their ID."""
        try:
            result = await self.users_collection.update_one(
                {
                    "_id": ObjectId(user_id),
                    "isDeleted": {"$ne": True}
                },
                {
                    "$set": {"password": new_password}
                }
            )
            if result.modified_count == 1:
                logger.info(f"Password updated for user {user_id}")
                return True
            else:
                logger.warning(f"No changes made to user {user_id} password.")
                return False
        except Exception as e:
            logger.error(f"Error updating password for user {user_id}: {e}")
            return False
