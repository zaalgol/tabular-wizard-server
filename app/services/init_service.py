from app.services.user_service import UserService
from app.config.config import Config 

class InitService:
    def __init__(self, app):
        self.user_service = UserService()
        self.seed_admin_user(app)
        self.seed_quest_user(app)

    def seed_admin_user(self, app):
        with app.app_context():
            email = Config.EMAIL_DOMAIN
            password=Config.ADMIN_PASSWORD
            return self.user_service.create_user(email, password)
    
    def seed_quest_user(self, app):
        with app.app_context():
            email = Config.EMAIL_DOMAIN
            password= Config.QUEST_PASSWORD
            return self.user_service.create_user(email, password)
        

