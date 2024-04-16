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


class TrainingTask:
    def __init__(self) -> None:
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()

    def run_task(self, model, headers, df, task_callback, app_context):
        is_training_successfully_finished = False
        trained_model = None
        evaluations = None
        try:
            if model.model_type == 'classification':
                training_model = LightgbmClassifier(train_df = df, prediction_column = model.target_column)
                evaluate = self.classificationEvaluate

            elif model.model_type == 'regression':
                training_model = LightGBMRegressor(train_df = df, prediction_column = model.target_column)
                evaluate = self.regressionEvaluate

            if model.training_speed == 'slow':
                model.tune_hyper_parameters()

            trained_model = training_model.train()
            evaluations = evaluate.evaluate_train_and_test(trained_model, training_model)
            print(evaluate.format_train_and_test_evaluation(evaluations))
            is_training_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            task_callback(model, trained_model, evaluations, headers, is_training_successfully_finished, app_context)


    
