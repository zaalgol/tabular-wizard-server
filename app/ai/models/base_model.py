from abc import abstractmethod
import pprint
import time
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import optuna
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.nlp_embeddings_preprocessing import NlpEmbeddingsPreprocessing


class BaseModel:
    def __init__(self, target_column, scoring):
        self.study = None
        self.target_column = target_column
        self.scoring = scoring
        self.data_preprocessing = DataPreprocessing()
            
    @property
    def default_values(self):
        return {}

    @property
    def callbacks(self):
        # Optuna handles time limits via study.optimize() parameters
        return []

    @property
    @abstractmethod
    def default_params(self):
        return {}
