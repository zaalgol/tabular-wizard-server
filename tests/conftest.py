import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import and initialize the Flask app
from app.app import create_app
app = create_app()

# Import the required modules and objects
from app.ai.models.classification.ensemble.ensemble import Ensemble
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from app.services.model_service import ModelService
from app.entities.model import Model
from app.app import socketio

# Make them available to your test files
__all__ = ['app', 'Ensemble', 'BaseClassfierModel', 'ModelService', 'Model', 'socketio']