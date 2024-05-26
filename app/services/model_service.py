import os
from app.ai.data_preprocessing import DataPreprocessing
import app.app as app
from datetime import datetime, UTC
from app.entities.model import Model
from app.repositories.model_repository import ModelRepository
from app.repositories.user_repository import UserRepository
from flask import current_app, jsonify, make_response, send_from_directory, send_from_directory, url_for, send_file
from werkzeug.utils import safe_join
from werkzeug.utils import secure_filename
import pandas as pd
from app.storage.model_storage import ModelStorage
from app.tasks.inference_task import InferenceTask
from app.tasks.training_task import TrainingTask
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
import threading
import pickle

# socketio = SocketIO(cors_allowed_origins="*")
from app.app import socketio


class ModelService:
    _instance = None

    def __init__(self):
        self.model_repository = ModelRepository()
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.training_task = TrainingTask()
        self.inference_task = InferenceTask()
        self.model_storage = ModelStorage()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def train_model(self, model, dataset):
        if dataset is None:
            return {"error": "No dataset provided"}, 400
        df = self.__dataset_to_df(dataset)
        app_context = current_app._get_current_object().app_context()

        thread = threading.Thread(target=self.training_task.run_task, args=(model, df.columns.tolist(), df, self.__training_task_callback, app_context))
        thread.start()
        socketio.emit('status', {'status': 'success', 'message': f'Model {model.model_name} training in process.'})
        
        return {}, 200, {}

    def __perprocess_data(self, df, drop_other_columns=None):
        if drop_other_columns:
            df = self.data_preprocessing.exclude_other_columns(df, columns=drop_other_columns)
        return df
    
    def __dataset_to_df(self, dataset):
        headers = dataset[0]
        data_rows = dataset[1:]
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    
    def inference(self, user_id, model_name, file_name, dataset):
        loaded_model =  self.model_storage.load_model(user_id, model_name)
        model_details_dict =  self.get_user_model_by_user_id_and_model_name(user_id, model_name)
        model_details = Model(**model_details_dict)
        model_details.user_id = user_id
        model_details.model_name = model_name
        model_details.file_name = file_name
        original_df = self.__dataset_to_df(dataset)
        original_df = self.__perprocess_data(original_df, drop_other_columns=model_details.columns)

        
        app_context = current_app._get_current_object().app_context()

        thread = threading.Thread(target=self.inference_task.run_task, args=(model_details, loaded_model, original_df, self.__inference_task_callback, app_context))
        thread.start()

    def __training_task_callback(self, model, trained_model, encoding_rules, transformations, headers, is_training_successfully_finished, app_context):
        try:
            with app_context:
                if not is_training_successfully_finished:
                    # Emit an event for training failure
                    socketio.emit('status', {'status': 'failed', 'message': f'Model {model.model_name} training failed.'})
                else:
                    saved_model_file_path = self.model_storage.save_model(trained_model, model.user_id, model.model_name)
                    model.encoding_rules = encoding_rules
                    model.transformations = transformations

                    self.model_repository.add_or_update_model_for_user(model, headers, saved_model_file_path)
                    
                    evaluations_url = self.__generate_model_evaluations_file(model)
                    socketio.emit('status', {'status': 'success',
                                            'file_type': 'evaluations',
                                            'model_name': f'{model.model_name}',
                                            'message': f'Model {model.model_name} training completed successfully.',
                                            'file_url': evaluations_url})
        except Exception as e:
            print(f"Error downloading file: {e}")
            
            socketio.emit('status', {'status': 'failed', 'message': f'Model {model.model_name} training failed.'})
            
    def __generate_model_evaluations_file(self, model):
        # Emit an event for training success
        SAVED_MODEL_FOLDER = os.path.join(app.Config.SAVED_MODELS_FOLDER, model.user_id, model.model_name)
        evaluations_filename = f"{model.model_name}__evaluations.txt"
        evaluations_filepath = os.path.join(SAVED_MODEL_FOLDER, evaluations_filename)
        
        if not os.path.exists(SAVED_MODEL_FOLDER):
            os.makedirs(SAVED_MODEL_FOLDER)
            
        with open(evaluations_filepath, 'w') as file:
            file.write(str(f"Model Name: {model.model_name}\nModel Type: {model.model_type} \nTraining Srategy: {model.training_strategy}\nSampling Strategy: {model.sampling_strategy}\n\nEvaluations:\n{model.evaluations}"))
            
        # Generate a unique URL for the txt file
        scheme = 'https' if current_app.config.get('PREFERRED_URL_SCHEME', 'http') == 'https' else 'http'
        server_name = current_app.config.get('SERVER_NAME', 'localhost:8080')
        return f"{scheme}://{server_name}/download/{evaluations_filename}"

    def __inference_task_callback(self, model_details, original_df, is_inference_successfully_finished, app_context):
        with app_context:
            if not is_inference_successfully_finished:
                # Emit an event for training failure
                socketio.emit('status', {'status': 'failed', 'message': f'Model {model_details.model_name} inference failed.'})
            else:
                # TODO: Add logs to DB
                current_utc_datetime = datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')
                SAVED_INFERENCES_FOLDER = os.path.join(app.Config.SAVED_INFERENCES_FOLDER, model_details.user_id, model_details.model_name)
                uploaf_file_without_sufix = model_details.file_name[:model_details.file_name.index(".")]
                csv_filename = f"{current_utc_datetime}__{model_details.model_name}__{uploaf_file_without_sufix}__inference.csv"
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
                    'file_type': 'inference',
                    'model_name': f'{model_details.model_name}',
                    'message': f'Model {model_details.model_name} inference completed successfully.',
                    'file_url': csv_url
                })
    
    def download_file(self, user_id, model_name, filename, file_type):
        try:
            if file_type == 'inference':
                saved_folder = current_app.config['SAVED_INFERENCES_FOLDER']
            else: 
                saved_folder = current_app.config['SAVED_MODELS_FOLDER']
            file_directory = safe_join(saved_folder, user_id, model_name)
            file_path = safe_join(os.getcwd(), file_directory, filename)
            
            if not os.path.isfile(file_path):
                current_app.logger.error(f"File not found: {file_path}")
                return jsonify({"msg": "File not found"}), 404

            current_app.logger.info(f"Serving file {file_path}")
            return send_file(file_path, as_attachment=True)
        except Exception as e:
            current_app.logger.error(f"Error downloading file: {e}")
            return jsonify({"msg": str(e)}), 500
        

    def get_user_models_by_id(self, user_id):
           result = self.model_repository.get_user_models_by_id(user_id, additonal_properties=['created_at', 'description'])
           return result
    
    def get_user_model_by_user_id_and_model_name(self, user_id, model_name):
        return self.model_repository.get_user_model_by_user_id_and_model_name(user_id, model_name,
                                                                                additonal_properties=['created_at', 'description', 'columns',
                                                                                                      'encoding_rules', 'transformations', 'metric', 'target_column',
                                                                                                      'model_type', 'training_strategy', 'sampling_strategy'])
        
    def generate_model_metric_file(self, user_id, model_name):
        try:
            model = self.model_repository.get_user_model_by_user_id_and_model_name(user_id, model_name,
                                                                                    additonal_properties=['evaluations',
                                                                                                        'model_type', 'training_strategy', 'sampling_strategy'])
            model_object = Model(**model)
            model_object.user_id = user_id
            model_object.model_name = model_name
            evaluations_url = self.__generate_model_evaluations_file(model_object)
                        
            socketio.emit('status', {'status': 'success',
                                    'file_type': 'evaluations',
                                    'model_name': f'{model_object.model_name}',
                                    'message': f'Model {model_object.model_name} evaluations download successfully.',
                                    'file_url': evaluations_url})
        except Exception as e:
            print(f"Error downloading file: {e}")
            
            socketio.emit('status', {'status': 'failed', 'message': f'Model {model_name} evaluations download failed.'})
        
    
    def delete_model_of_user(self, user_id, model_name):
        self.model_storage.delete_model(user_id, model_name)
        return self.model_repository.delete_model_of_user(user_id, model_name, hard_delete=True)
    
