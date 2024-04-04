import os
import app
from app.models.ai_model import AiModel
from app.repositories.ai_model_repository import AiModelRepository
from app.repositories.user_repository import UserRepository
from flask import current_app, jsonify, make_response
import pandas as pd
from tabularwizard import DataPreprocessing, LightgbmClassifier, LightGBMRegressor, ClassificationEvaluate, RegressionEvaluate
import threading
import pickle


class AiModelService:
    _instance = None

    def __init__(self):
        self.ai_model_repository = AiModelRepository()
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.classificationEvaluate = ClassificationEvaluate()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def train_model(self, ai_model, dataset):
        if dataset is None:
            return {"error": "No dataset provided"}, 400
        #df.to_csv('before.csv')
        df = self.perprocess_data(dataset)
        
        # df.to_csv('after.csv')

        # Capture the app context here
        app_context = current_app._get_current_object().app_context()

        thread = threading.Thread(target=self.training_task, args=(ai_model,  df.columns.tolist(), df, self.training_task_callback, app_context))
        thread.start()

    def perprocess_data(self, dataset, drop_other_columns=None):
        headers = dataset[0]
        data_rows = dataset[1:]
        df = pd.DataFrame(data_rows, columns=headers)
        if drop_other_columns:
            self.data_preprocessing.exclude_other_columns(df,columns=drop_other_columns)
        df = self.data_preprocessing.fill_missing_not_numeric_cells(df)
        df = self.data_preprocessing.fill_missing_numeric_cells(df)
        df = self.data_preprocessing.fill_missing_not_numeric_cells(df)
        df = self.data_preprocessing.sanitize_column_names(df)
        cat_features  =  self.data_preprocessing.get_all_categorical_columns_names(df)
        for feature in cat_features:
            df[feature] = df[feature].astype('category')
        df = df.set_index(headers[0])
        return df
    
    def inference(self, user_id, model_name, dataset):
        loaded_model = self.load_model(user_id, model_name)
        model_details_dict =  self.get_user_model_by_user_id_and_model_name(user_id, model_name)
        model_details = AiModel(**model_details_dict)
        df = self.perprocess_data(dataset, drop_other_columns=model_details.columns)
        X_data = self.data_preprocessing.exclude_columns(df, columns_to_exclude=model_details.target_column).copy()
        

        if model_details.model_type == 'classification':
            y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
            print(self.classificationEvaluate.evaluate_classification(df[model_details.target_column], y_predict))
        elif model_details.model_type == 'regression':
            y_predict = self.RegressionEvaluate.predict(loaded_model, X_data)
            print(self.RegressionEvaluate.evaluate_classification(df[model_details.target_column], y_predict))

    
    def training_task(self, ai_model, headers, df, training_task_callback, app_context):
        is_training_successfully_finish = False
        trained_model = None
        try:
            if ai_model.model_type == 'classification':
                model = LightgbmClassifier(train_df = df.copy(), prediction_column = ai_model.target_column)
            elif ai_model.model_type == 'regression':
                model = LightGBMRegressor(train_df = df.copy(), prediction_column = ai_model.target_column)
            
            if ai_model.training_speed == 'slow':
                model.tune_hyper_parameters()

            trained_model = model.train()
            is_training_successfully_finish = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            training_task_callback(ai_model, trained_model, headers, is_training_successfully_finish, app_context)


    def training_task_callback(self, ai_model, trained_model, headers, is_training_successfully_finish, app_context):
        with app_context:
            if not is_training_successfully_finish:
                # TODO: handle the exception
                pass
            else:
                saved_model_file_path = self.save_model(trained_model, ai_model.user_id, ai_model.model_name)
                self.ai_model_repository.add_or_update_ai_model_for_user(ai_model, headers, saved_model_file_path)

    def load_model(self, user_id, model_name):
        SAVED_MODEL_FOLDER = os.path.join(app.config.config.Config.SAVED_MODELS_FOLDER, user_id, model_name)
        SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'model.sav')
        if not os.path.exists(SAVED_MODEL_FOLDER):
            raise Exception(f"Model {SAVED_MODEL_FILE} not found")
        return pickle.load(open(SAVED_MODEL_FILE, 'rb'))

    def save_model(self, model, user_id, model_name):
            SAVED_MODEL_FOLDER = os.path.join(app.config.config.Config.SAVED_MODELS_FOLDER, user_id, model_name)
            SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'model.sav')
            if not os.path.exists(SAVED_MODEL_FOLDER):
                os.makedirs(SAVED_MODEL_FOLDER)
            pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))
            return SAVED_MODEL_FILE

    def get_user_ai_models_by_id(self, user_id):
           return self.ai_model_repository.get_user_ai_models_by_id(user_id, additonal_properties=['created_at', 'description'])
    
    def get_user_model_by_user_id_and_model_name(self, user_id, model_name):
        return self.ai_model_repository.get_user_model_by_user_id_and_model_name(user_id, model_name,
                                                                                  additonal_properties=['created_at', 'description', 'columns', 'target_column', 'model_type', 'training_speed'])[0]




        



