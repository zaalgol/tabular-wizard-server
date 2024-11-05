# app/services/init_service.py

from app.services.user_service import UserService
from app.config.config import Config

class InitService:
    def __init__(self, app):
        self.app = app
        self.db = app.state.db  # Access the database from app state
        self.user_service = UserService(self.db)
    
    def seed_admin_user(self):
        email = Config.ADMIN_EMAIL  # Use appropriate config variables
        password = Config.ADMIN_PASSWORD
        existing_user = self.user_service.user_repository.get_user_by_email(email)
        if existing_user:
            print(f"Admin user {email} already exists.")
            return existing_user
        else:
            print(f"Creating admin user {email}.")
            return self.user_service.create_user(email, password)
    
    def seed_quest_user(self):
        email = Config.QUEST_EMAIL
        password = Config.QUEST_PASSWORD
        existing_user = self.user_service.user_repository.get_user_by_email(email)
        if existing_user:
            print(f"Quest user {email} already exists.")
            return existing_user
        else:
            print(f"Creating quest user {email}.")
            return self.user_service.create_user(email, password)
