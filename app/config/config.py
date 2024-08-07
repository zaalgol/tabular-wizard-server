import os
from dotenv import load_dotenv

# Load the .env file
dotenv_path = '/tabular-wizard-server/.env'  # Adjust the path as needed
load_dotenv(dotenv_path)

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL') # use for multi Docker container deployment
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///mydatabase.db'  # Use SQLite for local development
    MONGODB_URI = os.getenv('MONGODB_URI')
    IS_MONGO_LOCAL = os.getenv('IS_MONGO_LOCAL', 1)
    IS_STORAGE_LOCAL = os.getenv('IS_STORAGE_LOCAL', 1)
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
    QUEST_PASSWORD = os.getenv('QUEST_PASSWORD')
    EMAIL_DOMAIN = os.getenv('EMAIL_DOMAIN')
    JWT_HEADER_NAME = JWT_QUERY_STRING_NAME = 'Authorization'
    JWT_ACCESS_TOKEN_EXPIRES = os.getenv('JWT_ACCESS_TOKEN_EXPIRES')
    SAVED_MODELS_FOLDER = os.getenv('SAVED_MODELS_FOLDER')
    SAVED_INFERENCES_FOLDER = os.getenv('SAVED_INFERENCES_FOLDER')
    SERVER_NAME = os.getenv('SERVER_NAME')
    
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    MODEL=os.getenv('MODEL')
    MAX_TOKENS=os.getenv('MAX_TOKENS')
    LLM_MAX_TRIES=os.getenv('LLM_MAX_TRIES')
    
