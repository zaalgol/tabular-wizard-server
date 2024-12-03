import os
from dotenv import load_dotenv

# Load the .env file
# dotenv_path = '/tabular-wizard-server/.env'  # Adjust the path as needed
# dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')

load_dotenv(dotenv_path)

class Config:
    DEBUG_MODE=1

    ACCESS_TOKEN_SECRET_KEY  = os.getenv('ACCESS_TOKEN_SECRET_KEY')
    REFRESH_TOKEN_SECRET_KEY  = os.getenv('REFRESH_TOKEN_SECRET_KEY')
    ALGORITHM = os.getenv('ALGORITHM')
    ACCESS_TOKEN_EXPIRE_MINUTES= int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'))
    REFRESH_TOKEN_EXPIRE_DAYS= int(os.getenv('REFRESH_TOKEN_EXPIRE_DAYS'))
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL') # use for multi Docker container deployment
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///mydatabase.db'  # Use SQLite for local development
    MONGODB_URI = os.getenv('MONGODB_URI')
    IS_MONGO_LOCAL = os.getenv('IS_MONGO_LOCAL', 1)
    IS_STORAGE_LOCAL = os.getenv('IS_STORAGE_LOCAL', 1)
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')
    QUEST_EMAIL = os.getenv('QUEST_EMAIL')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
    QUEST_PASSWORD = os.getenv('QUEST_PASSWORD')

    JWT_HEADER_NAME = JWT_QUERY_STRING_NAME = 'Authorization'
    JWT_ACCESS_TOKEN_EXPIRES = os.getenv('JWT_ACCESS_TOKEN_EXPIRES')
    SAVED_MODELS_FOLDER = os.getenv('SAVED_MODELS_FOLDER')
    SAVED_INFERENCES_FOLDER = os.getenv('SAVED_INFERENCES_FOLDER')
    SERVER_NAME = os.getenv('SERVER_NAME')
    
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    
    DATASET_SPLIT_SIZE = os.getenv('DATASET_SPLIT_SIZE', 0.3)

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    MODEL=os.getenv('MODEL')
    MAX_TOKENS=os.getenv('MAX_TOKENS')
    LLM_MAX_TRIES=os.getenv('LLM_MAX_TRIES')
    LLM_NUMBER_OF_DATASET_LINES=os.getenv('LLM_NUMBER_OF_DATASET_LINES')

    CSV_URL_PREFIX=os.getenv('CSV_URL_PREFIX')
    
