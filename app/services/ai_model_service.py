import os
import app
from datetime import datetime, UTC
from app.models.ai_model import AiModel
from app.repositories.ai_model_repository import AiModelRepository
from app.repositories.user_repository import UserRepository
from flask import current_app, jsonify, make_response, send_from_directory, send_from_directory, url_for, send_file
from werkzeug.utils import safe_join
from werkzeug.utils import secure_filename
import pandas as pd
from tabularwizard import DataPreprocessing, LightgbmClassifier, LightGBMRegressor, ClassificationEvaluate, RegressionEvaluate
import threading
import pickle

from flask_socketio import SocketIO

# socketio = SocketIO(cors_allowed_origins="*")
from app import socketio


class AiModelService:
    _instance = None

    def __init__(self):
        self.ai_model_repository = AiModelRepository()
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def train_model(self, ai_model, dataset):
        if dataset is None:
            return {"error": "No dataset provided"}, 400
        df = self._dataset_to_df(dataset)
        df = self._perprocess_data(df, target_column=ai_model.target_column)
       
        # df.to_csv('after.csv')

        # Capture the app context here
        app_context = current_app._get_current_object().app_context()

        thread = threading.Thread(target=self.training_task, args=(ai_model,  df.columns.tolist(), df, self._training_task_callback, app_context))
        thread.start()

    def _perprocess_data(self, df, target_column=None, drop_other_columns=None):
        
        if drop_other_columns:
            df = self.data_preprocessing.exclude_other_columns(df, columns=drop_other_columns)

        # if target_column: 
        #     df = self.data_preprocessing.convert_column_categircal_values_to_numerical_values(df, target_column)

        # df = self.data_preprocessing.one_hot_encode_all_categorical_columns(df)    
        # columns_to_encode = df.columns[df.columns != target_column]
        # df = self.data_preprocessing.one_hot_encode_all_categorical_columns(df, columns_to_encode)
        # df = self.data_preprocessing.one_hot_encode_column(df, 'color')
        # df = self.data_preprocessing.convert_column_categircal_values_to_numerical_values(df, 'type')
        # df = self.data_preprocessing.fill_missing_numeric_cells(df)
        # df = self.data_preprocessing.sanitize_column_names(df)

        # TODO: find a resample methos tha works with categorical columns
        cat_features  =  self.data_preprocessing.get_all_categorical_columns_names(df)
        for feature in cat_features:
            df[feature] = df[feature].astype('category')
        return df
    
    def _dataset_to_df(self, dataset):
        headers = dataset[0]
        data_rows = dataset[1:]
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    
    def inference(self, user_id, model_name, dataset):
        loaded_model = self.load_model(user_id, model_name)
        model_details_dict =  self.get_user_model_by_user_id_and_model_name(user_id, model_name)
        model_details = AiModel(**model_details_dict)
        model_details.user_id = user_id
        model_details.model_name = model_name
        original_df = self._dataset_to_df(dataset)
        original_df = self._perprocess_data(original_df, drop_other_columns=model_details.columns)
        X_data = self.data_preprocessing.exclude_columns(original_df, columns_to_exclude=model_details.target_column).copy()
        
        app_context = current_app._get_current_object().app_context()

        thread = threading.Thread(target=self.inference_task, args=(model_details, loaded_model, original_df, X_data, self._inference_task_callback, app_context))
        thread.start()

        # if model_details.model_type == 'classification':
        #     y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
        #     print(self.classificationEvaluate.evaluate_classification(df[model_details.target_column], y_predict))
        # elif model_details.model_type == 'regression':
        #     y_predict = self.RegressionEvaluate.predict(loaded_model, X_data)
        #     print(self.RegressionEvaluate.evaluate_classification(df[model_details.target_column], y_predict))

    def inference_task(self, model_details, loaded_model, original_df, X_data, inference_task_callback, app_context):
        try:
            is_inference_successfully_finished = False
            if model_details.model_type == 'classification':
                y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
            elif model_details.model_type == 'regression':
                y_predict = self.RegressionEvaluate.predict(loaded_model, X_data)
            original_df[f'{model_details.target_column}_predict'] = y_predict
            is_inference_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            inference_task_callback(model_details, original_df, is_inference_successfully_finished, app_context)

    def training_task(self, ai_model, headers, df, training_task_callback, app_context):
        is_training_successfully_finished = False
        trained_model = None
        evaluations = None
        try:
            if ai_model.model_type == 'classification':
                model = LightgbmClassifier(train_df = df, prediction_column = ai_model.target_column)
                evaluate = self.classificationEvaluate

            elif ai_model.model_type == 'regression':
                model = LightGBMRegressor(train_df = df, prediction_column = ai_model.target_column)
                evaluate = self.regressionEvaluate

            if ai_model.training_speed == 'slow':
                model.tune_hyper_parameters()

            trained_model = model.train()
            evaluations = evaluate.evaluate_train_and_test(trained_model, model)
            evaluate.print_train_and_test_evaluation(evaluations)
            is_training_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            training_task_callback(ai_model, trained_model, evaluations, headers, is_training_successfully_finished, app_context)


    def _training_task_callback(self, ai_model, trained_model, evaluations, headers, is_training_successfully_finished, app_context):
        with app_context:
            if not is_training_successfully_finished:
                # Emit an event for training failure
                socketio.emit('status', {'status': 'failed', 'message': f'Model {ai_model.model_name} training failed.'})
            else:
                saved_model_file_path = self.save_model(trained_model, ai_model.user_id, ai_model.model_name)
                self.ai_model_repository.add_or_update_ai_model_for_user(ai_model, evaluations, headers, saved_model_file_path)
                # Emit an event for training success
                socketio.emit('status', {'status': 'success', 'message': f'Model {ai_model.model_name} training completed successfully.'})

    def _inference_task_callback(self, model_details, original_df, is_inference_successfully_finished, app_context):
        with app_context:
            if not is_inference_successfully_finished:
                # Emit an event for training failure
                socketio.emit('status', {'status': 'failed', 'message': f'Model {model_details.model_name} inference failed.'})
            else:
                # TODO: Add logs to DB
                current_utc_datetime = datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')
                SAVED_INFERENCES_FOLDER = os.path.join(app.config.config.Config.SAVED_INFERENCES_FOLDER, model_details.user_id, model_details.model_name)
                csv_filename = f"{current_utc_datetime}_{model_details.model_name}_inference.csv"
                csv_filepath = os.path.join(SAVED_INFERENCES_FOLDER, csv_filename)
                if not os.path.exists(SAVED_INFERENCES_FOLDER):
                    os.makedirs(SAVED_INFERENCES_FOLDER)
                original_df.to_csv(csv_filepath, index=False)

                # Generate a unique URL for the CSV file
                scheme = 'https' if current_app.config.get('PREFERRED_URL_SCHEME', 'http') == 'https' else 'http'
                server_name = current_app.config.get('SERVER_NAME', 'localhost:8080')
                csv_url = f"{scheme}://{server_name}/download/{csv_filename}"

                # Emit event with the URL to the CSV file
                socketio.emit('status', {
                    'status': 'success',
                    'model_name': f'{model_details.model_name}',
                    'message': f'Model {model_details.model_name} inference completed successfully.',
                    'csv_url': csv_url
                })

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
    
    def downloadInferenceFile(self, user_id, model_name, filename):
        try:
            saved_inferences_folder = current_app.config['SAVED_INFERENCES_FOLDER']
            file_directory = safe_join(saved_inferences_folder, user_id, model_name)
            file_path = safe_join(os.getcwd(), file_directory, filename)
            
            if not os.path.isfile(file_path):
                current_app.logger.error(f"File not found: {file_path}")
                return jsonify({"msg": "File not found"}), 404

            current_app.logger.info(f"Serving file {file_path}")
            return send_file(file_path, as_attachment=True)
        except Exception as e:
            current_app.logger.error(f"Error downloading file: {e}")
            return jsonify({"msg": str(e)}), 500
        

    def get_user_ai_models_by_id(self, user_id):
           return self.ai_model_repository.get_user_ai_models_by_id(user_id, additonal_properties=['created_at', 'description'])
    
    def get_user_model_by_user_id_and_model_name(self, user_id, model_name):
        return self.ai_model_repository.get_user_model_by_user_id_and_model_name(user_id, model_name,
                                                                                  additonal_properties=['created_at', 'description', 'columns', 'target_column', 'model_type', 'training_speed'])




        


