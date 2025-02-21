import os
import pickle
import shutil
import app.app as app


class LocalModelStorage:
    
    def load_model(self, user_id, model_name):
        SAVED_MODEL_FOLDER = os.path.join(app.Config.SAVED_MODELS_FOLDER, user_id, model_name)
        SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'model.sav')
        if not os.path.exists(SAVED_MODEL_FOLDER):
            raise Exception(f"Model {SAVED_MODEL_FILE} not found")
        return pickle.load(open(SAVED_MODEL_FILE, 'rb'))

    def save_model(self, model, user_id, model_name):
            SAVED_MODEL_FOLDER = os.path.join(app.Config.SAVED_MODELS_FOLDER, user_id, model_name)
            SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'model.sav')
            if not os.path.exists(SAVED_MODEL_FOLDER):
                os.makedirs(SAVED_MODEL_FOLDER)
            pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))
            return SAVED_MODEL_FILE
        
    def delete_model(self, user_id, model_name):
        SAVED_MODEL_FOLDER = os.path.join(app.Config.SAVED_MODELS_FOLDER, user_id, model_name)
        if not os.path.exists(SAVED_MODEL_FOLDER):
            return f"Model {model_name} for user {user_id} does not exist in storage"
            # raise Exception(f"Model {model_name} for user {user_id} not found")
        shutil.rmtree(SAVED_MODEL_FOLDER)
        return f"Model {model_name} for user {user_id} has been deleted from storage"