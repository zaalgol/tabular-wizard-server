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
    
    def add_or_update_ai_model_for_user(self, user_id, model_name, description, saved_model_file_path):

        # Define the field paths using dot notation
        model_field_path = f"ai_models.{model_name}.filePath"
        created_at_field_path = f"ai_models.{model_name}.created_at"
        description_field_path = f"ai_models.{model_name}.description"
        
        # Get the current UTC datetime
        current_utc_datetime = datetime.now(UTC)
        
        # Update the user document with the model path and current UTC datetime
        update_result = self.users_collection.update_one(
            {"_id": ObjectId(user_id), "isDeleted": {"$ne": True}},
            {
                "$set": {
                    model_field_path: saved_model_file_path,
                    description_field_path: description,
                    created_at_field_path: current_utc_datetime
                }
            }
        )
    
    def get_user_ai_models_by_id(self, user_id):
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
            return result.get("ai_models", {})
        else:
            return {}


    
