#https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'learning_rate': (0.01, 0.3, 'log-uniform'),  # typical range from learning rate
    'num_leaves': (31, 200, 'int'),  # depends on max_depth, should be smaller than 2^(max_depth)
    'max_depth': (3, 11, 'int'),  # typical values can range from 3-10
    'min_child_samples': (10, 200, 'int'),  # minimum number of data needed in a child (leaf)
    'min_child_weight': (1e-5, 1e-3, 'log-uniform'),  # deals with under-fitting
    'subsample': (0.5, 1.0, 'uniform'),  # commonly used range
    'subsample_freq': (1, 10, 'int'),  # frequency for bagging
    'colsample_bytree': (0.5, 1.0, 'uniform'),  # fraction of features that can be selected for each tree
    'reg_alpha': (1e-9, 20.0, 'log-uniform'),  # L1 regularization term
    'reg_lambda': (1e-9, 20.0, 'log-uniform'),  # L2 regularization term
    'n_estimators': (50, 1000, 'int'),  # number of boosted trees to fit
}

class LightGBMRegressor(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df=train_df, target_column=target_column, *args, **kwargs)
        
        self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        self.X_test = DataPreprocessing().set_not_numeric_as_categorial(self.X_test)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LGBMRegressor( verbosity=-1,*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
  