import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from src.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh', 'identity'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.05, 0.1, 0.5, 1, 2, 3, 4],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 300, 500],
    'learning_rate_init': [0.001, 0.01, 0.05],
}

class MLPNetRegressor(BaseRegressorModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False, 
                 create_transformations=False, apply_transformations=False, test_size=0.3, already_splitted_data=None, scoring='r2',
                 *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,  
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                         create_transformations=create_transformations, apply_transformations=apply_transformations,
                         already_splitted_data=already_splitted_data, scoring=scoring, *args, **kwargs)
        self.estimator = MLPRegressor(*args, **kwargs)

    def train(self):
            if self.search: # with hyperparameter tuning
                result = self.search.fit(self.X_train, self.y_train)
                print("Best Cross-Validation parameters:", self.search.best_params_)
                print("Best Cross-Validation score:", self.search.best_score_)
            else:
                result = self.estimator.fit(self.X_train, self.y_train)
            return result
        

    def tune_hyper_parameters(self, params=None, kfold=10):
            if params is None:
                params = self.default_params
            Kfold = KFold(n_splits=kfold)  
            
            self.search = GridSearchCV(estimator=self.estimator,
                                        param_grid=params,
                                        scoring=self.scoring,
                                        n_jobs=1, 
                                        cv=Kfold,
                                        verbose=0)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
