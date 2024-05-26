import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

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
    def __init__(self, train_df, target_column, hidden_layer_sizes=None,
                 *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        
        if not hidden_layer_sizes:
            first_layer_size=max(len(self.X_train.columns), 2)
            second_layer_size=max(int(first_layer_size /2), 2)
            hidden_layer_sizes=(first_layer_size, second_layer_size)
        self.estimator = MLPRegressor(max_iter=500, hidden_layer_sizes=hidden_layer_sizes, *args, **kwargs)

    def train(self):
            if self.search: # with hyperparameter tuning
                result = self.search.fit(self.X_train, self.y_train)
                print("Best Cross-Validation parameters:", self.search.best_params_)
                print("Best Cross-Validation score:", self.search.best_score_)
            else:
                result = self.estimator.fit(self.X_train, self.y_train)
            return result
        

    def tune_hyper_parameters(self, params=None, kfold=10, *args, **kwargs):
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
