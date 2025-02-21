import os
import asyncio
from datetime import datetime, timezone

import pandas as pd
from fastapi import HTTPException
from pymongo.database import Database

from app.config.config import Config 
from app.entities.model import Model
from app.services.report_file_service import ReportFileService
from app.services.websocket_service import WebsocketService
from app.repositories.model_repository import ModelRepository
from app.storage.local_model_storage import LocalModelStorage
from app.storage.model_storage import ModelStorage
from app.ai.tasks.inference_task import InferenceTask
from app.ai.tasks.training_task import TrainingTask
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
from app.ai.data_preprocessing import DataPreprocessing


class ModelService:
    _instance = None

    def __init__(self, app, db: Database):
        self.model_repository = ModelRepository(db)
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.reportFileService = ReportFileService()
        self.websocketService = WebsocketService(app)
        self.training_task = TrainingTask()
        self.inference_task = InferenceTask()
        self.csv_url_prefix = Config.CSV_URL_PREFIX

        if int(Config.IS_STORAGE_LOCAL):
            self.model_storage = LocalModelStorage()
        else:
            self.model_storage = ModelStorage()

    def __new__(cls, app, db: Database):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def train_model(self, model: Model, dataset):
        await self.websocketService.emit("status", {"status": "in-process"}, user_id=model.user_id)

        if dataset is None:
            raise HTTPException(status_code=400, detail="No dataset provided")
        
        model.file_line_num = len(dataset)
        df = self.__dataset_to_df(dataset)

        model.status = "training"
        headers = df.columns.tolist()
        # We pass None for saved_model_file_path if there's no model file yet:
        self.model_repository.add_or_update_model_for_user(model, headers, saved_model_file_path=None)

        await self.websocketService.emit(
            "status",
            {
                "status": "success",
                "message": f"Model {model.model_name} training in process."
            },
            user_id=model.user_id
        )

        result = await self.__run_training_task(model, df)
        return result

    async def __run_training_task(self, model: Model, df: pd.DataFrame):
        headers = df.columns.tolist()
        trained_model, error_msg = await self.__execute_in_background_or_not(
            self.training_task.run_task, model, df
        )

        if error_msg:
            return await self.__handle_training_failure(model, headers, error_msg)
        return await self.__handle_training_success(model, headers, trained_model, df)

    async def __execute_in_background_or_not(self, func, *args, **kwargs):
        """
        Helper that executes a function either in the event loop (if DEBUG_MODE)
        or in a background thread.
        """
        if Config.DEBUG_MODE:
            return func(*args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    async def __handle_training_failure(self, model: Model, headers: list, error_msg: str):
        """
        Handles the logic when the training task fails.
        """
        model.status = "fail"
        message = f'Model {model.model_name} training failed with error: "{error_msg}"'
        self.model_repository.add_or_update_model_for_user(model, headers, None)
        await self.websocketService.emit(
            'status',
            {'status': 'failed', 'message': message},
            user_id=model.user_id
        )
        return {'status': 'failed', 'message': message}

    async def __handle_training_success(self, model: Model, headers: list, trained_model, df: pd.DataFrame):
        """
        Handles the logic when the training task succeeds.
        """
        saved_model_file_path = self.model_storage.save_model(
            trained_model, model.user_id, model.model_name
        )
        model.model_description_pdf_file_path = await self.reportFileService.generate_model_evaluations_file(
            model, df.copy()
        )
        model.status = "success"
        self.model_repository.add_or_update_model_for_user(model, headers, saved_model_file_path)
        
        await self.websocketService.emit(
            'status',
            {
                'status': 'success',
                'file_type': 'evaluations',
                'model_name': f'{model.model_name}',
                'message': f'Model {model.model_name} training completed successfully.',
                'file_url': model.model_description_pdf_file_path
            },
            user_id=model.user_id
        )

        return {
            'status': 'success',
            'message': f'Model {model.model_name} training completed successfully.',
            'file_url': model.model_description_pdf_file_path
        }

    def __remove_columns_not_in_train_dataset(self, df, drop_other_columns=None):
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
        model_details_dict = await self.get_user_model_by_user_id_and_model_name(user_id, model_name)
        model_details = Model(**model_details_dict)
        model_details.user_id = user_id
        model_details.model_name = model_name
        model_details.file_name = file_name

        await self.websocketService.emit(
            'status',
            {
                'status': 'success',
                'message': f'Model {model_details.model_name} inference in process.'
            },
            user_id=user_id
        )

        inference_df = self.__dataset_to_df(dataset)
        inference_df = self.__remove_columns_not_in_train_dataset(
            inference_df, drop_other_columns=model_details.columns
        )

        result = await self.__run_inference_task(model_details, loaded_model, inference_df)
        return result

    async def __run_inference_task(self, model_details: Model, loaded_model, inference_df: pd.DataFrame):
        model_details, inference_df, success = await self.__execute_in_background_or_not(
            self.inference_task.run_task, model_details, loaded_model, inference_df
        )

        if not success:
            return await self.__handle_inference_failure(model_details)
        return await self.__handle_inference_success(model_details, inference_df)

    async def __handle_inference_failure(self, model_details: Model):
        """
        Handles logic when the inference task fails.
        """
        await self.websocketService.emit(
            'status',
            {
                'status': 'failed',
                'message': f'Model {model_details.model_name} inference failed.'
            },
            user_id=model_details.user_id
        )
        return {
            'status': 'failed',
            'message': f'Model {model_details.model_name} inference failed.'
        }

    async def __handle_inference_success(self, model_details: Model, inference_df: pd.DataFrame):
        """
        Handles logic when the inference task succeeds, including saving
        results to CSV and emitting success messages.
        """
        current_utc_datetime = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
        saved_folder = Config.SAVED_INFERENCES_FOLDER
        saved_inferences_folder = os.path.join(saved_folder, model_details.user_id, model_details.model_name)

        # Extract the file name without extension
        upload_file_without_suffix = model_details.file_name[:model_details.file_name.index(".")]
        csv_filename = (
            f"{current_utc_datetime}__{model_details.model_name}__"
            f"{upload_file_without_suffix}__inference.csv"
        )
        csv_filepath = os.path.join(saved_inferences_folder, csv_filename)
        
        if not os.path.exists(saved_inferences_folder):
            os.makedirs(saved_inferences_folder)
        inference_df.to_csv(csv_filepath, index=False)

        csv_url = f"/download/{csv_filename}"

        await self.websocketService.emit(
            'status',
            {
                'status': 'success',
                'file_type': 'inference',
                'model_name': f'{model_details.model_name}',
                'message': f'Model {model_details.model_name} inference completed successfully.',
                'file_url': self.csv_url_prefix + csv_url
            },
            user_id=model_details.user_id
        )

        return {
            'status': 'success',
            'message': f'Model {model_details.model_name} inference completed successfully.',
            'file_url': csv_url
        }

    async def download_file(self, user_id, model_name, filename, file_type):
        try:
            if file_type == 'inference':
                saved_folder = Config.SAVED_INFERENCES_FOLDER
            else: 
                saved_folder = Config.SAVED_MODELS_FOLDER
            return await self.reportFileService.download_file(
                user_id, model_name, filename, saved_folder
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_user_models_by_id(self, user_id):
        return await self.model_repository.get_user_models_by_id(
            user_id, 
            additional_properties=[
                'created_at', 'description', 'metric', 'train_score', 'test_score', 
                'target_column','model_type', 'training_strategy', 'sampling_strategy',
                'is_multi_class','file_line_num', 'file_name', 'sampling_strategy', 
                'status'
            ]
        )
    
    async def get_user_model_by_user_id_and_model_name(self, user_id, model_name):
        return await self.model_repository.get_user_model_by_user_id_and_model_name(
            user_id, 
            model_name, 
            additional_properties=[
                'created_at', 'description', 'columns', 'columns_type', 'embedding_rules',
                'encoding_rules','transformations', 'metric', 'target_column','model_type', 
                'training_strategy','sampling_strategy', 'is_multi_class','is_time_series', 
                'time_series_code', 'formated_evaluations', 'status'
            ]
        )
        
    async def get_model_details_file(self, user_id, model_name):
        try:
            model = await self.model_repository.get_user_model_by_user_id_and_model_name(
                user_id, 
                model_name, 
                additional_properties=['model_description_pdf_file_path']
            )
            
            await self.websocketService.emit('status', {
                'status': 'success',
                'file_type': 'evaluations',
                'model_name': f'{model_name}',
                'message': f'Model {model_name} evaluations download successfully.',
                'file_url': model['model_description_pdf_file_path']
            })
        except Exception as e:
            print(f"Error during model details file retrieval: {e}")
            self.websocketService.emit(
                'status', 
                {'status': 'failed', 'message': f'Model {model_name} evaluations download failed.'}
            )
        
    async def delete_model_of_user(self, user_id, model_name):
        self.model_storage.delete_model(user_id, model_name)
        return await self.model_repository.delete_model_of_user(user_id, model_name, hard_delete=True)
