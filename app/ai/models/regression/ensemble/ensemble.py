# https://chat.openai.com/c/60d698a6-9405-4db9-8831-6b2491bb9111
from typing import OrderedDict
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import cross_val_score
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel
from app.ai.models.regression.implementations.linear_regression import LinearRegressorModel
from app.ai.models.regression.implementations.random_forest_regressor import RandomForestRegressorModel
from app.ai.models.regression.implementations.svr_regressor import SVRRegressorModel
from app.ai.models.regression.implementations.catboot_regressor import CatboostRegressor
from app.ai.models.regression.evaluate import Evaluate
from app.ai.models.regression.implementations.lightgbm_regerssor import LightGBMRegressor
from app.ai.models.regression.implementations.mlrpregressor import MLPNetRegressor
from app.ai.models.regression.implementations.xgboost_regressor import XgboostRegressor
from sklearn.ensemble import VotingRegressor
from itertools import islice

class Ensemble(BaseRegressorModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                  create_transformations=False, apply_transformations=False, test_size=0.3, scoring='neg_root_mean_squared_error', top_n_best_models=3):
        self.regressors = {}
        super().__init__(train_df=train_df, target_column=target_column, scoring=scoring, split_column=split_column, test_size=test_size,
                    create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                    create_transformations=create_transformations, apply_transformations=apply_transformations)
        self.already_splitted_data = {'X_train': self.X_train, 'X_test': self.X_test, 'y_train': self.y_train, 'y_test':self.y_test}
        self.evaluate = Evaluate()
        self.scoring = scoring
        self.top_n_best_models = top_n_best_models

    def create_models(self, df):
        model_classes = {
            'lgbm_regressor': LightGBMRegressor,
            'mlr_regressor': MLPNetRegressor,
            'xgb_regressor': XgboostRegressor,
            'rf_regressor': RandomForestRegressorModel,
            'svr_regressor': SVRRegressorModel,
            'cat_regressor': CatboostRegressor,
            'linear_regressor': LinearRegressorModel
        }
        self.regressors = {
            key: self._regressor_factory(model_class, df)
            for key, model_class in model_classes.items()
        }
       
    def _regressor_factory(self, model_class, train_df):
        return {
        'model': model_class(
                train_df=train_df.copy(), 
                target_column=self.target_column, 
                already_splitted_data=self.already_splitted_data,
                scoring=self.scoring
            )
        }
        
    def tune_hyper_parameters(self):
        for regressor_value in self.regressors.values():
            regressor_value['model'].tune_hyper_parameters()
            
    def tuning_top_models(self):
        top_models = list(islice(self.regressors.items(), self.top_n_best_models))
        for name, model_info in top_models:
            print(f"Tuning and retraining {name}...")
            model_info['model'].tune_hyper_parameters()
            model_info['trained_model'] = model_info['model'].train()
            model_info['evaluations'] = self.evaluate.evaluate_train_and_test(model_info['trained_model'], model_info['model'])

    def train_all_models(self):
        for regressor_value in self.regressors.values():
            regressor_value['trained_model'] = regressor_value['model'].train()

    def sort_models_by_score(self):
        
        scores = {name: cross_val_score(value['model'].estimator, self.X_train, self.y_train, cv=5, scoring=self.scoring) 
                  for name, value in self.regressors.items()}
        average_scores = {name: score.mean() for name, score in scores.items()}
        print(f"average cross validation scores {average_scores}")
        sorted_names = sorted(average_scores, key=average_scores.get, reverse=True)# self.scoring=='neg_mean_absolute_error')
        self.regressors = OrderedDict((name, self.regressors[name]) for name in sorted_names)

    def create_voting_regressor(self):
        model_list = [(name, info['model'].estimator) for name, info in islice(self.regressors.items(), self.top_n_best_models)]
        self.voting_regressor = VotingRegressor(estimators=model_list)

    def train_voting_regressor(self):
        self.trained_voting_regressor = self.voting_regressor.fit(self.X_train, self.y_train)

    def evaluate_voting_regressor(self):
        self.voting_regressor_evaluations = self.evaluate.evaluate_train_and_test(self.trained_voting_regressor, self)
