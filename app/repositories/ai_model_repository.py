from flask import current_app

class AiModelRepository:
    @property
    def models_collection(self):
        return self.db['aiModels']