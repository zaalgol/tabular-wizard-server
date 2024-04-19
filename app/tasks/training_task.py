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
from tabularwizard import DataPreprocessing, LightgbmClassifier, LightGBMRegressor, ClassificationEvaluate, RegressionEvaluate, KnnClassifier, Ensemble
import threading
import pickle

# socketio = SocketIO(cors_allowed_origins="*")
from app import socketio


class TrainingTask:
    def __init__(self) -> None:
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.data_preprocessing = DataPreprocessing()

    def run_task(self, model, headers, df, task_callback, app_context):
        is_training_successfully_finished = False
        trained_model = None
        evaluations = None
        encoding_rules = None
        try:
            if model.ensemble:
                trained_model, evaluations, encoding_rules= self._train_multi_models(model, df)
            else:
                trained_model, evaluations, encoding_rules = self._train_single_model(model, df)
                # if model.model_type == 'classification':
                #     training_model = LightgbmClassifier(train_df = df, prediction_column = model.target_column)
                #     evaluate = self.classificationEvaluate

                # elif model.model_type == 'regression':
                #     training_model = LightGBMRegressor(train_df = df, prediction_column = model.target_column)
                #     evaluate = self.regressionEvaluate

                # if model.training_speed == 'slow':
                #     model.tune_hyper_parameters(model.metric)

                # trained_model = training_model.train()
                # evaluations = evaluate.evaluate_train_and_test(trained_model, training_model)
                # print(evaluate.format_train_and_test_evaluation(evaluations))
            is_training_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            task_callback(model, trained_model, encoding_rules, evaluations, headers, is_training_successfully_finished, app_context)

    def _train_single_model(self, model, df):
        if model.model_type == 'classification':
            training_model = LightgbmClassifier(train_df = df, prediction_column = model.target_column)
            evaluate = self.classificationEvaluate

        elif model.model_type == 'regression':
            training_model = LightGBMRegressor(train_df = df, prediction_column = model.target_column)
            evaluate = self.regressionEvaluate

        if model.training_speed == 'slow':
            model.tune_hyper_parameters(model.metric)

        trained_model = training_model.train()
        evaluations = evaluate.evaluate_train_and_test(trained_model, training_model)
        print(evaluate.format_train_and_test_evaluation(evaluations))
        return trained_model, evaluations, None
        

    def _train_multi_models(self, model, df):
        if model.model_type == 'classification':
            df = self._data_preprocessing(df, model.target_column)
            ensemble = Ensemble(train_df = df, prediction_column = model.target_column, create_encoding_rules=True, apply_encoding_rules=True)
            ensemble.create_models(df)
            ensemble.train_all_models()
            ensemble.evaluate_all_models()

            ensemble.create_voting_classifier()
            ensemble.train_voting_classifier()
            ensemble.evaluate_voting_classifier()

            evaluate = self.classificationEvaluate
            print(evaluate.format_train_and_test_evaluation(ensemble.voting_classifier_evaluations))
            return ensemble.trained_voting_classifier, ensemble.voting_classifier_evaluations, ensemble.encoding_rules
        
    def _data_preprocessing(self, df, target_column=None):
        df_copy=df.copy()
          # df = self.data_preprocessing.one_hot_encode_all_categorical_columns(df)    
        # columns_to_encode = df.columns[df.columns != target_column]
        # df = self.data_preprocessing.fill_missing_not_numeric_cells(df)
        data_preprocessing = DataPreprocessing()
        df_copy = data_preprocessing.sanitize_dataframe(df_copy)
        df = data_preprocessing.fill_missing_numeric_cells(df_copy)
        # encoding_rules = data_preprocessing.create_encoding_rules(df_copy)
        # df_copy = data_preprocessing.apply_encoding_rules(df_copy, encoding_rules)
        # df = self.data_preprocessing.one_hot_encode_column(df, 'color')
        # df = self.data_preprocessing.convert_column_categircal_values_to_numerical_values(df, 'type')
        # df = self.data_preprocessing.fill_missing_numeric_cells(df)
        # df = self.data_preprocessing.sanitize_column_names(df)
        return df_copy #, encoding_rules 


    
