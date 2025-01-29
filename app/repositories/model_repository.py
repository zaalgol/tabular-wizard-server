from bson import ObjectId
from fastapi import Depends
from datetime import datetime, timezone
from pymongo.database import Database

class ModelRepository:
    def __init__(self, db: Database):
        self._db = db

    @property
    def db(self):
        return self._db
    
    @property
    def users_collection(self):
        return self.db['users']
    
    def add_or_update_model_for_user(self, model, columns, saved_model_file_path):
        model_name = model.model_name
        file_name_path = f"models.{model_name}.file_name"
        file_line_num_path = f"models.{model_name}.file_line_num"
        model_field_path = f"models.{model_name}.filePath"
        created_at_field_path = f"models.{model_name}.created_at"
        description_field_path = f"models.{model_name}.description"
        columns_field_path = f"models.{model_name}.columns"
        columns_type_field_path = f"models.{model_name}.columns_type"
        embedding_rules_field_path = f"models.{model_name}.embedding_rules"
        target_column_field_path = f"models.{model_name}.target_column"
        model_type_field_path = f"models.{model_name}.model_type"
        training_strategy_field_path = f"models.{model_name}.training_strategy"
        sampling_strategy_field_path = f"models.{model_name}.sampling_strategy"
        formated_field_path = f"models.{model_name}.evaluations"
        formated_evaluations_field_path = f"models.{model_name}.formated_evaluations"
        metric_field_path = f"models.{model_name}.metric"
        encoding_rules_field_path = f"models.{model_name}.encoding_rules"
        transformations_field_path = f"models.{model_name}.transformations"
        isDeleted_field_path = f"models.{model_name}.isDeleted"
        is_multi_class_field_path = f"models.{model_name}.is_multi_class"
        train_score_column_field_path = f"models.{model_name}.train_score"
        test_score_column_field_path = f"models.{model_name}.test_score"
        model_description_pdf_file_path_path = f"models.{model_name}.model_description_pdf_file_path"
        is_time_series_field_path = f"models.{model_name}.is_time_series"
        time_series_code_field_path = f"models.{model_name}.time_series_code"
        # columns_type
        # Get the current UTC datetime
        current_utc_datetime = datetime.now(timezone.utc)
        
        # Update the user document with the model path and current UTC datetime
        return self.users_collection.update_one(
            {"_id": ObjectId(model.user_id)},
            {
                "$set": {
                    file_name_path: model.file_name,
                    file_line_num_path: model.file_line_num,
                    model_field_path: saved_model_file_path,
                    description_field_path: model.description,
                    created_at_field_path: current_utc_datetime,
                    columns_type_field_path: model.columns_type,
                    columns_field_path: columns,
                    embedding_rules_field_path: model.embedding_rules,
                    encoding_rules_field_path: model.encoding_rules,
                    target_column_field_path: model.target_column,
                    model_type_field_path: model.model_type,
                    training_strategy_field_path: model.training_strategy,
                    sampling_strategy_field_path: model.sampling_strategy,
                    metric_field_path: model.metric,
                    formated_field_path: model.evaluations,
                    formated_evaluations_field_path: model.formated_evaluations,
                    transformations_field_path: model.transformations,
                    isDeleted_field_path: False,
                    is_multi_class_field_path: model.is_multi_class,
                    train_score_column_field_path: model.train_score,
                    test_score_column_field_path: model.test_score,
                    model_description_pdf_file_path_path: model.model_description_pdf_file_path,
                    is_time_series_field_path: model.is_time_series,
                    time_series_code_field_path: model.time_series_code
                }
            }
        )
    
    async def get_user_model_by_user_id_and_model_name(self, user_id, model_name, additional_properties):
        pipeline = [
            {"$match": {"_id": ObjectId(user_id), "isDeleted": {"$ne": True}}},
            {"$project": {
            model_name: f"$models.{model_name}",
            "_id": 0  # Exclude the _id from the results if not needed
        }},
            {"$match": {"specific_model.isDeleted": {"$ne": True}}}  # Ensure the model is not marked as deleted
        ]

        result = await self.users_collection.aggregate(pipeline).next()
        if result:
            return self._model_dict_to_front_list(result, additional_properties)[0]
        else:
            return {}

    
    async def get_user_models_by_id(self, user_id, additional_properties):
        pipeline = [
            {"$match": {"_id": ObjectId(user_id), "isDeleted": {"$ne": True}}},
            {"$project": {"models": {"$objectToArray": "$models"}}},
            {"$addFields": {"models": {"$filter": {
                "input": "$models",
                "cond": {"$not": "$$this.v.isDeleted"}
            }}}},
            {"$project": {"models": {"$arrayToObject": "$models"}}}
        ]
        result = await self.users_collection.aggregate(pipeline).next()
        if result and result["models"]:
            return self._model_dict_to_front_list(result.get("models", {}), additional_properties)
        else:
            return {}
        
    async def delete_model_of_user(self, user_id, model_name, hard_delete=False):
        """
        Delete a model for a user. 
        If hard_delete is True, delete the model physically from the database.
        Otherwise, set its 'isDeleted' field to True.
        """
        model_field_path = f"models.{model_name}"
        if hard_delete:
            result = await self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$unset": {model_field_path: ""}}
            )
        else:
            result = await self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {f"{model_field_path}.isDeleted": True}}
            )
        return result
        

    def _model_dict_to_front_list(self, models_dict, additional_properties):
        models_list = []
        for name, details in models_dict.items():
            # Initialize model_info with the id
            model_info = {'id': name}
            # Dynamically add properties from additional_properties if they exist in details
            for property in additional_properties:
                if property in details:
                    model_info[property] = details[property]
            models_list.append(model_info)
        return models_list
