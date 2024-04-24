import os
import app
from datetime import datetime, UTC
from app.entities.model import Model
from app.repositories.model_repository import ModelRepository
from app.repositories.user_repository import UserRepository
from flask import current_app, jsonify, make_response, send_from_directory, send_from_directory, url_for, send_file
from werkzeug.utils import safe_join
from werkzeug.utils import secure_filename
import pandas as pd
from tabularwizard import DataPreprocessing, LightgbmClassifier, LightGBMRegressor, ClassificationEvaluate, RegressionEvaluate, KnnClassifier
import threading
import pickle

# socketio = SocketIO(cors_allowed_origins="*")
from app import socketio


class InferenceTask:
    def __init__(self) -> None:
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()

    def run_task(self, model_details, loaded_model, original_df, X_data, inference_task_callback, app_context):
        try:
            is_inference_successfully_finished = False
            X_data = self._data_preprocessing(X_data, model_details.encoding_rules)
            
            if model_details.model_type == 'classification':
                y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
            elif model_details.model_type == 'regression':
                y_predict = self.regressionEvaluate.predict(loaded_model, X_data)
            original_df[f'{model_details.target_column}_predict'] = y_predict
            is_inference_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            inference_task_callback(model_details, original_df, is_inference_successfully_finished, app_context)

    def _data_preprocessing(self, df, encoding_rules):
        df_copy = df.copy()
        df_copy = self.data_preprocessing.sanitize_dataframe(df_copy)
        df_copy = self.data_preprocessing.fill_missing_numeric_cells(df_copy)
        df_copy = self.data_preprocessing.set_not_numeric_as_categorial(df)
        if encoding_rules:
            df_copy = self.data_preprocessing.apply_encoding_rules(df_copy, encoding_rules)
        return df_copy


