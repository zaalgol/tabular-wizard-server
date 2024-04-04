from bson import ObjectId
from flask import current_app
from datetime import datetime, UTC
from flask import current_app

class AiModelRepository:
    def __init__(self):
        self.current_app = current_app

    @property
    def db(self):
        return self.current_app.db
    
    @property
    def users_collection(self):
        return self.db['users']
    
    def add_or_update_ai_model_for_user(self, ai_model, columns, saved_model_file_path):
        model_name = ai_model.model_name
        # Define the field paths using dot notation
        model_field_path = f"ai_models.{model_name}.filePath"
        created_at_field_path = f"ai_models.{model_name}.created_at"
        description_field_path = f"ai_models.{model_name}.description"
        columns_field_path = f"ai_models.{model_name}.columns"
        target_column_field_path = f"ai_models.{model_name}.target_column"
        model_type_field_path = f"ai_models.{model_name}.model_type"
        training_speed_field_path = f"ai_models.{model_name}.training_speed"
        
        # Get the current UTC datetime
        current_utc_datetime = datetime.now(UTC)
        
        # Update the user document with the model path and current UTC datetime
        update_result = self.users_collection.update_one(
            {"_id": ObjectId(ai_model.user_id), "isDeleted": {"$ne": True}},
            {
                "$set": {
                    model_field_path: saved_model_file_path,
                    description_field_path: ai_model.description,
                    created_at_field_path: current_utc_datetime,
                    columns_field_path: columns,
                    target_column_field_path: ai_model.target_column,
                    model_type_field_path: ai_model.model_type,
                    training_speed_field_path: ai_model.training_speed
                }
            }
        )
    
    def get_user_model_by_user_id_and_model_name(self, user_id, model_name, additonal_properties):
        pipeline = [
            {"$match": {"_id": ObjectId(user_id), "isDeleted": {"$ne": True}}},
            {"$project": {
            model_name: f"$ai_models.{model_name}",
            "_id": 0  # Exclude the _id from the results if not needed
        }},
            {"$match": {"specific_model.isDeleted": {"$ne": True}}}  # Ensure the model is not marked as deleted
        ]

        result =  self.users_collection.aggregate(pipeline).next()
        if result:
            return self._model_dict_to_front_list(result, additonal_properties)
        else:
            return {}

    
    def get_user_ai_models_by_id(self, user_id, additonal_properties):

        # pipeline = [
        #     {"$match": {"_id": ObjectId(user_id), "isDeleted": {"$ne": True}}},
        #     {"$project": {
        #         "specific_model": f"$ai_models.with_columns_and_target_column",
        #         "_id": 0  # Exclude the _id from the results if not needed
        #     }},
        #     {"$match": {"specific_model.isDeleted": {"$ne": True}}}  # Ensure the model is not marked as deleted
        # ]

        # result =  self.users_collection.aggregate(pipeline).next()
    

        pipeline = [
            {"$match": {"_id": ObjectId(user_id), "isDeleted": {"$ne": True}}},
            {"$project": {"ai_models": {"$objectToArray": "$ai_models"}}},
            {"$addFields": {"ai_models": {"$filter": {
                "input": "$ai_models",
                "cond": {"$not": "$$this.v.isDeleted"}
            }}}},
            {"$project": {"ai_models": {"$arrayToObject": "$ai_models"}}}
        ]
        result = self.users_collection.aggregate(pipeline).next()
        if result:
            return self._model_dict_to_front_list(result.get("ai_models", {}), additonal_properties)
        else:
            return {}
        
    # def _model_dict_to_front_list(self, models_dict):
    #     models_list = []
    #     for name, details in models_dict.items():
    #         model_info = {'id': name}
    #         if 'created_at' in details:
    #             model_info['created_at'] = details['created_at']
    #         if 'description' in details:
    #             model_info['description'] = details['description']
    #         models_list.append(model_info)
    #     return models_list

    def _model_dict_to_front_list(self, models_dict, additonal_properties):
        models_list = []
        for name, details in models_dict.items():
            # Initialize model_info with the id
            model_info = {'id': name}
            # Dynamically add properties from additonal_properties if they exist in details
            for property in additonal_properties:
                if property in details:
                    model_info[property] = details[property]
            models_list.append(model_info)
        return models_list
    
