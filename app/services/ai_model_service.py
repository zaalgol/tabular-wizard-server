import app
from app.repositories.ai_model_repository import AiModelRepository
from app.repositories.user_repository import UserRepository
from flask import jsonify, make_response
import pandas as pd
from tabularwizard import DataPreprocessing, LightgbmClassifier, LightGBMRegressor



class AiModelService:
    _instance = None

    def __init__(self):
        self.user_repository = UserRepository()
        
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def train_model(self, user_id, modelName, headers, data_rows, target_column, model_type, training_speed):
        df = pd.DataFrame(data_rows, columns=headers)
        df.to_csv('before.csv')
        self.perprocess_data(df)
        df = df.set_index(headers[0])
        df.to_csv('after.csv')
        if model_type == 'classification':
            lightgbmClassifier = LightgbmClassifier(train_df = df.copy(), prediction_column = target_column)
            lightgbmClassifier.tune_hyper_parameters()
            lightgbmClassifier.train()
            pass
        elif model_type == 'regression':
            lightGBMRegressor = LightGBMRegressor(train_df = df.copy(), prediction_column = target_column)
            lightGBMRegressor.tune_hyper_parameters()
            lightGBMRegressor.train()

    def perprocess_data(self, df):
        data_preprocessing = DataPreprocessing()
        df = data_preprocessing.fill_missing_not_numeric_cells(df)
        df = data_preprocessing.fill_missing_numeric_cells(df)
        df = data_preprocessing.fill_missing_not_numeric_cells(df)
        df = data_preprocessing.sanitize_column_names(df)
        cat_features  =  data_preprocessing.get_all_categorical_columns_names(df)
        for feature in cat_features:
            df[feature] = df[feature].astype('category')
        return df


        



