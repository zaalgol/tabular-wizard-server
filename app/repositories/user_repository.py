from app import db
import app
from app.utils.utils import with_session
from app.models.user import User
from sqlalchemy import and_, func, asc, desc, String

class UserRepository:
    @with_session
    def get_user_by_username(self, username, session=None):
        return session.query(User).filter(and_(User.username == str(username), User.isDeleted.is_(None))).first()
    
    @with_session
    def get_user_by_email(self, email, session=None):
        return session.query(User).filter(and_(User.email == str(email), User.isDeleted.is_(None))).first()
    
    @with_session
    def get_user_by_email_and_password(self, email, password, session=None):
        # self.seed_admin_user()
        temp = session.query(User).first()
        return session.query(User).filter(and_(User.email == str(email),
                                                User.password == str(password),
                                                User.isDeleted.is_(None))).first()

    @with_session
    def create_user(self, username, password, session=None):
        user = User(username=username, password=password)  # Assume password hashing
        session.add(user)
        session.commit()
        return user
    
    @with_session
    def seed_admin_user(self, session=None):
        admin_exists = User.query.filter_by(email='admin').first()
        if not admin_exists:
            admin = User(email='admin', password=app.config.config.Config.ADMIN_PASSWORD)  # Use a hashed password in real scenarios
            db.session.add(admin)
            db.session.commit()
            print('Admin user created')
        else:
            print('Admin user already exists')