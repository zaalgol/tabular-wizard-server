import os
import asyncio
from datetime import datetime, timezone
from app.ai.data_preprocessing import DataPreprocessing
from app.app import get_app
from app.entities.model import Model
from app.repositories.model_repository import ModelRepository
from app.config.config import Config 
from fastapi import Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pymongo.database import Database
from werkzeug.utils import safe_join
import pandas as pd
from app.storage.local_model_storage import LocalModelStorage
from app.storage.model_storage import ModelStorage
from app.tasks.inference_task import InferenceTask
from app.tasks.training_task import TrainingTask
from app.services.report_file_service import ReportFileService
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
import matplotlib.pyplot as plt
from fastapi_socketio import SocketManager

plt.switch_backend('Agg')

class ModelService:
    _instance = None

    def __init__(self, db: Database):
        # self.config = Config
        self.model_repository = ModelRepository(db)
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.reportFileService = ReportFileService()
        self.training_task = TrainingTask()
        self.inference_task = InferenceTask()
        self.socketio = get_app().state.socketio 
        self.csv_url_prefix = f"http://localhost:8080"

        if int(Config.IS_STORAGE_LOCAL):
            self.model_storage = LocalModelStorage()
        else:
            self.model_storage = ModelStorage()

    def __new__(cls, db: Database):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # def train_model(self, model: Model, dataset, background_tasks: BackgroundTasks):
    async def train_model(self, model: Model, dataset):
        if dataset is None:
            raise HTTPException(status_code=400, detail="No dataset provided")
        
        model.file_line_num = len(dataset)
        df = self.__dataset_to_df(dataset)

        await self.socketio.emit('status', {'status': 'success', 'message': f'Model {model.model_name} training in process.'})
        result = await self.__run_training_task(model, df)

        return result

        # background_tasks.add_task(self.__run_training_task, model, df)

        # self.socketio.emit('status', {'status': 'success', 'message': f'Model {model.model_name} training in process.'})
        # return JSONResponse(content={}, status_code=200)

    # def __run_training_task(self, model, df):
    #     self.training_task.run_task(model, df.columns.tolist(), df, self.__training_task_callback)
    async def __run_training_task(self, model, df):
        result = await asyncio.to_thread(self.training_task.run_task, model, df.columns.tolist(), df)
        df, model, trained_model, encoding_rules, transformations, headers, is_training_successfully_finished = result

        # df, model, trained_model, encoding_rules, transformations, headers, is_training_successfully_finished = \
        #     self.training_task.run_task(model, df.columns.tolist(), df)


        if not is_training_successfully_finished:
            await self.socketio.emit('status', {'status': 'failed', 'message': f'Model {model.model_name} training failed.'})
            return {'status': 'failed', 'message': f'Model {model.model_name} training failed.'}
        else:
            saved_model_file_path = self.model_storage.save_model(trained_model, model.user_id, model.model_name)
            model.encoding_rules = encoding_rules
            model.transformations = transformations

            model.model_description_pdf_file_path = self.reportFileService.generate_model_evaluations_file(model, df.copy())
            
            self.model_repository.add_or_update_model_for_user(model, headers, saved_model_file_path)
            
            await self.socketio.emit('status', {
                'status': 'success',
                'file_type': 'evaluations',
                'model_name': f'{model.model_name}',
                'message': f'Model {model.model_name} training completed successfully.',
                'file_url': model.model_description_pdf_file_path
            })

            return {
                'status': 'success',
                'message': f'Model {model.model_name} training completed successfully.',
                'file_url': model.model_description_pdf_file_path
            }

    def __preprocess_data(self, df, drop_other_columns=None):
        if drop_other_columns:
            df = self.data_preprocessing.exclude_other_columns(df, columns=drop_other_columns)
        return df
    
    def __dataset_to_df(self, dataset):
        headers = dataset[0]
        data_rows = dataset[1:]
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    
    async def inference(self, user_id, model_name, file_name, dataset):
        loaded_model = self.model_storage.load_model(user_id, model_name)
        model_details_dict = self.get_user_model_by_user_id_and_model_name(user_id, model_name)
        model_details = Model(**model_details_dict)
        model_details.user_id = user_id
        model_details.model_name = model_name
        model_details.file_name = file_name

        original_df = self.__dataset_to_df(dataset)
        original_df = self.__preprocess_data(original_df, drop_other_columns=model_details.columns)

        result = await self.__run_inference_task(model_details, loaded_model, original_df)

        return result

    async def __run_inference_task(self, model_details, loaded_model, original_df):
        result = await asyncio.to_thread(self.inference_task.run_task, model_details, loaded_model, original_df)
        model_details, original_df, is_inference_successfully_finished = result

        if not is_inference_successfully_finished:
            await self.socketio.emit('status', {'status': 'failed', 'message': f'Model {model_details.model_name} inference failed.'})
            return {'status': 'failed', 'message': f'Model {model_details.model_name} inference failed.'}
        else:
            current_utc_datetime = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
            saved_folder = Config.SAVED_INFERENCES_FOLDER
            saved_inferences_folder = os.path.join(saved_folder, model_details.user_id, model_details.model_name)
            upload_file_without_suffix = model_details.file_name[:model_details.file_name.index(".")]
            csv_filename = f"{current_utc_datetime}__{model_details.model_name}__{upload_file_without_suffix}__inference.csv"
            csv_filepath = os.path.join(saved_inferences_folder, csv_filename)
            
            if not os.path.exists(saved_inferences_folder):
                os.makedirs(saved_inferences_folder)
            original_df.to_csv(csv_filepath, index=False)

            csv_url = f"/download/{csv_filename}"
            
            await self.socketio.emit('status', {
                'status': 'success',
                'file_type': 'inference',
                'model_name': f'{model_details.model_name}',
                'message': f'Model {model_details.model_name} inference completed successfully.',
                'file_url': self.csv_url_prefix  + csv_url
            })

            return {
                'status': 'success',
                'message': f'Model {model_details.model_name} inference completed successfully.',
                'file_url': csv_url
            }

    def __training_task_callback(self, df, model, trained_model, encoding_rules, transformations, headers, is_training_successfully_finished):
        try:
            if not is_training_successfully_finished:
                self.socketio.emit('status', {'status': 'failed', 'message': f'Model {model.model_name} training failed.'})
            else:
                saved_model_file_path = self.model_storage.save_model(trained_model, model.user_id, model.model_name)
                model.encoding_rules = encoding_rules
                model.transformations = transformations

                model.model_description_pdf_file_path = self.reportFileService.generate_model_evaluations_file(model, df.copy())
                
                self.model_repository.add_or_update_model_for_user(model, headers, saved_model_file_path)
                
                self.socketio.emit('status', {
                    'status': 'success',
                    'file_type': 'evaluations',
                    'model_name': f'{model.model_name}',
                    'message': f'Model {model.model_name} training completed successfully.',
                    'file_url': model.model_description_pdf_file_path
                })
        except Exception as e:
            print(f"Error during training task callback: {e}")
            self.socketio.emit('status', {'status': 'failed', 'message': f'Model {model.model_name} training failed.'})
    
    # def __inference_task_callback(self, model_details, original_df, is_inference_successfully_finished):
    #     if not is_inference_successfully_finished:
    #         self.socketio.emit('status', {'status': 'failed', 'message': f'Model {model_details.model_name} inference failed.'})
    #     else:
    #         current_utc_datetime = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    #         saved_folder = Config.SAVED_INFERENCES_FOLDER
    #         saved_inferences_folder = os.path.join(saved_folder, model_details.user_id, model_details.model_name)
    #         upload_file_without_suffix = model_details.file_name[:model_details.file_name.index(".")]
    #         csv_filename = f"{current_utc_datetime}__{model_details.model_name}__{upload_file_without_suffix}__inference.csv"
    #         csv_filepath = os.path.join(saved_inferences_folder, csv_filename)
            
    #         if not os.path.exists(saved_inferences_folder):
    #             os.makedirs(saved_inferences_folder)
    #         original_df.to_csv(csv_filepath, index=False)

    #         csv_url = f"/download/{csv_filename}"
            
    #         self.socketio.emit('status', {
    #             'status': 'success',
    #             'file_type': 'inference',
    #             'model_name': f'{model_details.model_name}',
    #             'message': f'Model {model_details.model_name} inference completed successfully.',
    #             'file_url': csv_url
    #         })
    
    def download_file(self, user_id, model_name, filename, file_type):
        try:
            if file_type == 'inference':
                saved_folder = Config.SAVED_INFERENCES_FOLDER
            else: 
                saved_folder = Config.SAVED_MODELS_FOLDER
            file_directory = safe_join(saved_folder, user_id, model_name)
            file_path = safe_join(os.getcwd(), file_directory, filename)
            
            if not os.path.isfile(file_path):
                raise HTTPException(status_code=404, detail="File not found")

            return FileResponse(file_path, filename=filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_user_models_by_id(self, user_id):
        return self.model_repository.get_user_models_by_id(user_id, additional_properties=[
            'created_at', 'description', 'metric', 'train_score', 'test_score', 'target_column',
            'model_type', 'training_strategy', 'sampling_strategy', 'is_multi_class',
            'file_line_num', 'file_name', 'sampling_strategy'
        ])
    
    def get_user_model_by_user_id_and_model_name(self, user_id, model_name):
        return self.model_repository.get_user_model_by_user_id_and_model_name(user_id, model_name, additional_properties=[
            'created_at', 'description', 'columns', 'encoding_rules', 'transformations', 'metric', 'target_column',
            'model_type', 'training_strategy', 'sampling_strategy', 'is_multi_class',
            'is_time_series', 'time_series_code', 'formated_evaluations'
        ])
        
    async def get_model_details_file(self, user_id, model_name):
        try:
            model = self.model_repository.get_user_model_by_user_id_and_model_name(user_id, model_name, additional_properties=['model_description_pdf_file_path'])
            
            await self.socketio.emit('status', {
                'status': 'success',
                'file_type': 'evaluations',
                'model_name': f'{model_name}',
                'message': f'Model {model_name} evaluations download successfully.',
                'file_url': model['model_description_pdf_file_path']
            })
        except Exception as e:
            print(f"Error during model details file retrieval: {e}")
            self.socketio.emit('status', {'status': 'failed', 'message': f'Model {model_name} evaluations download failed.'})
        
    def delete_model_of_user(self, user_id, model_name):
        self.model_storage.delete_model(user_id, model_name)
        return self.model_repository.delete_model_of_user(user_id, model_name, hard_delete=True)
