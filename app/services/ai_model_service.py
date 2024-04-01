import os
import app
from app.repositories.ai_model_repository import AiModelRepository
from app.repositories.user_repository import UserRepository
from flask import current_app, jsonify, make_response
import pandas as pd
from tabularwizard import DataPreprocessing, LightgbmClassifier, LightGBMRegressor
import threading
import pickle


class AiModelService:
    _instance = None

    def __init__(self):
        self.ai_model_repository = AiModelRepository()
        
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def train_model(self, user_id, model_name, description, dataset, target_column, model_type, training_speed):
        if dataset is None:
            return {"error": "No dataset provided"}, 400
        headers = dataset[0]
        data_rows = dataset[1:]
        df = pd.DataFrame(data_rows, columns=headers)
        #df.to_csv('before.csv')
        df = self.perprocess_data(df)
        df = df.set_index(headers[0])
        # df.to_csv('after.csv')

        # Capture the app context here
        app_context = current_app._get_current_object().app_context()

        thread = threading.Thread(target=self.training_task, args=(model_type, training_speed,
                                                                    target_column, df, user_id, model_name, description,
                                                                    self.training_task_callback, app_context))
        thread.start()

    def perprocess_data(self, df):
        data_preprocessing = DataPreprocessing()
        df = data_preprocessing.fill_missing_not_numeric_cells(df)
        df = data_preprocessing.fill_missing_numeric_cells(df)
        df = data_preprocessing.fill_missing_not_numeric_cells(df)
        df = data_preprocessing.sanitize_column_names(df)
        cat_features  =  data_preprocessing.get_all_categorical_columns_names(df)
        for feature in cat_features:
            df[feature] = df[feature].astype('category')
        return df
    
    def training_task(self, model_type, training_speed, target_column, df, user_id, model_name, description, training_task_callback, app_context):
        is_training_successfully_finish = False
        trained_model = None
        try:
            if model_type == 'classification':
                model = LightgbmClassifier(train_df = df.copy(), prediction_column = target_column)
                pass
            elif model_type == 'regression':
                model = LightGBMRegressor(train_df = df.copy(), prediction_column = target_column)
            
            if training_speed == 'slow':
                model.tune_hyper_parameters()

            trained_model = model.train()
            is_training_successfully_finish = True
        finally:
            training_task_callback(trained_model, user_id, model_name, description, is_training_successfully_finish, app_context)


    def training_task_callback(self, model, user_id, model_Name, description, is_training_successfully_finish, app_context):
        with app_context:
            if not is_training_successfully_finish:
                # TODO: handle the exception
                pass
            else:
                saved_model_file_path = self.save_model(model, user_id, model_Name)
                self.ai_model_repository.add_or_update_ai_model_for_user(user_id, model_Name, description, saved_model_file_path)


    def save_model(self, model, user_id, model_name):
            SAVED_MODEL_FOLDER = os.path.join(app.config.config.Config.SAVED_MODELS_FOLDER, user_id, model_name)
            SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'model.sav')
            if not os.path.exists(SAVED_MODEL_FOLDER):
                os.makedirs(SAVED_MODEL_FOLDER)
            pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))
            return SAVED_MODEL_FILE

    def get_user_ai_models_by_id(self, user_id):
           return self.ai_model_repository.get_user_ai_models_by_id(user_id)




        



