import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL') # use for multi Docker container deployment
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///mydatabase.db'  # Use SQLite for local development
    MONGODB_URI = 'mongodb://localhost:27017/tabular-wizard'
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
    EMAIL_DOMAIN = os.getenv('EMAIL_DOMAIN')