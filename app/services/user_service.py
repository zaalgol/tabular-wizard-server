import app
from app.repositories.user_repository import UserRepository
from app.services.hassing_service import PasswordHasher
from app.services.token_serivce import TokenService
from flask import current_app, jsonify, make_response

import os
import pandas as pd
from tabularwizard import DataPreprocessing, LightgbmClassifier, LightGBMRegressor
import threading
import pickle

class UserService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.user_repository = UserRepository()
        self.token_service = TokenService()

    def login(self, email, password):
        # self.seed_admin_user() # TODO: find away to run migrations
        user = self.user_repository.get_user_by_email(email)
        if user:
            is_valid_password = PasswordHasher.check_password(user['password'], password)
            if is_valid_password:
                access_token = self.token_service.create_jwt_token(str(user['_id']))
                response = make_response(jsonify({"message": "Login successful", "access_token": access_token}), 200)
                return response
        return jsonify({'message': 'Invalid credentials'}), 401
    
    def seed_admin_user(self):
        email=f'admin@{app.config.config.Config.EMAIL_DOMAIN}'
        password=app.config.config.Config.ADMIN_PASSWORD
        return self.create_user(email, password)

    def create_user(self, email, password):
        hashed_password = PasswordHasher.hash_password(password)
        user = self.user_repository.create_user(email, hashed_password) 
        return user
    
    def get_user_by_id(self, user_id):
        return self.user_repository.get_user_by_id(user_id)
    

    # def train_model(self, user_id, modelName, headers, data_rows, target_column, model_type, training_speed):
    #     df = pd.DataFrame(data_rows, columns=headers)
    #     df.to_csv('before.csv')
    #     df = self.perprocess_data(df)
    #     df = df.set_index(headers[0])
    #     df.to_csv('after.csv')

    #     # Capture the app context here
    #     app_context = current_app._get_current_object().app_context()

    #     thread = threading.Thread(target=self.training_task, args=(model_type, training_speed,
    #                                                                 target_column, df, user_id, modelName,
    #                                                                 self.training_task_callback, app_context))
    #     thread.start()

    # def perprocess_data(self, df):
    #     data_preprocessing = DataPreprocessing()
    #     df = data_preprocessing.fill_missing_not_numeric_cells(df)
    #     df = data_preprocessing.fill_missing_numeric_cells(df)
    #     df = data_preprocessing.fill_missing_not_numeric_cells(df)
    #     df = data_preprocessing.sanitize_column_names(df)
    #     cat_features  =  data_preprocessing.get_all_categorical_columns_names(df)
    #     for feature in cat_features:
    #         df[feature] = df[feature].astype('category')
    #     return df
    
    # def training_task(self, model_type, training_speed, target_column, df, user_id, modelName, training_task_callback, app_context):
    #     is_training_successfully_finish = False
    #     trained_model = None
    #     try:
    #         if model_type == 'classification':
    #             model = LightgbmClassifier(train_df = df.copy(), prediction_column = target_column)
    #             pass
    #         elif model_type == 'regression':
    #             model = LightGBMRegressor(train_df = df.copy(), prediction_column = target_column)
            
    #         if training_speed == 'slow':
    #             model.tune_hyper_parameters()

    #         trained_model = model.train()
    #         is_training_successfully_finish = True
    #     finally:
    #         training_task_callback(trained_model, user_id, modelName, is_training_successfully_finish, app_context)


    # def training_task_callback(self, model, user_id, modelName, is_training_successfully_finish, app_context):
    #     with app_context:
    #         if not is_training_successfully_finish:
    #             # TODO: handle the exception
    #             pass
    #         else:
    #             saved_model_file_path = self.save_model(model, user_id, modelName)
    #             self.user_repository.add_or_update_ai_model_for_user(user_id, modelName, saved_model_file_path)


    # def save_model(self, model, user_id, modelName):
    #         SAVED_MODEL_FOLDER = os.path.join(app.config.config.Config.SAVED_MODELS_FOLDER, user_id, modelName)
    #         SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'model.sav')
    #         if not os.path.exists(SAVED_MODEL_FOLDER):
    #             os.makedirs(SAVED_MODEL_FOLDER)
    #         pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))
    #         return SAVED_MODEL_FILE
