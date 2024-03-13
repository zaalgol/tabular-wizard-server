from app import db
import app
from app.utils.utils import with_session
from app.models.user import User
from sqlalchemy import and_, func, asc, desc, String

class UserRepository:
    @with_session
    @staticmethod
    def get_user_by_username(username, session=None):
        return session.query(User).filter(and_(User.username == str(username), User.deleted.is_(None))).first()
    
    @with_session
    @staticmethod
    def get_user_by_email(email, session=None):
        return session.query(User).filter(and_(User.email == str(email), User.deleted.is_(None))).first()
    
    @with_session
    @staticmethod
    def get_user_by_email_and_password(email, password, session=None):
        return session.query(User).filter(and_(User.email == str(email),
                                                User.password == str(password),
                                                User.deleted.is_(None))).first()

    @with_session
    @staticmethod
    def create_user(username, password, session=None):
        user = User(username=username, password=password)  # Assume password hashing
        session.add(user)
        session.commit()
        return user
    
    @with_session
    @staticmethod
    def seed_admin_user():
        admin_exists = User.query.filter_by(username='admin').first()
        if not admin_exists:
            admin = User(username='admin', password=app.config['ADMIN_PASSWORD'])  # Use a hashed password in real scenarios
            db.session.add(admin)
            db.session.commit()
            print('Admin user created')
        else:
            print('Admin user already exists')