import boto3
import os
import pickle
from botocore.exceptions import NoCredentialsError
from app.config.config import Config

# AWS S3 configuration
AWS_ACCESS_KEY = Config.AWS_ACCESS_KEY
AWS_SECRET_KEY = Config.AWS_SECRET_KEY
BUCKET_NAME = Config.BUCKET_NAME

# Initialize S3 client
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

class ModelStorage:
    def load_model(self, user_id, model_name):
        # S3 Key for the model file
        SAVED_MODEL_KEY = f'models/{user_id}/{model_name}/model.sav'
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=SAVED_MODEL_KEY)
            model_data = response['Body'].read()
            return pickle.loads(model_data)
        except s3_client.exceptions.NoSuchKey:
            raise Exception(f"Model {SAVED_MODEL_KEY} not found in S3 bucket.")
        except NoCredentialsError:
            raise Exception("Credentials not available")

    def save_model(self, model, user_id, model_name):
        # S3 Key for the model file
        SAVED_MODEL_KEY = f'models/{user_id}/{model_name}/model.sav'
        try:
            model_data = pickle.dumps(model)
            s3_client.put_object(Body=model_data, Bucket=BUCKET_NAME, Key=SAVED_MODEL_KEY)
            return SAVED_MODEL_KEY
        except NoCredentialsError:
            raise Exception("Credentials not available")

# Example Usage
# model_storage = ModelStorage()
# model = ... # Your trained model
# model_storage.save_model(model, 'user123', 'mymodel')
# loaded_model = model_storage.load_model('user123', 'mymodel')
