#https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataPreprocessing
from src.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'learning_rate': (0.01, 0.3, 'log-uniform'),  # typical range from learning rate
    'num_leaves': (31, 200),  # depends on max_depth, should be smaller than 2^(max_depth)
    'max_depth': (3, 11),  # typical values can range from 3-10
    'min_child_samples': (10, 200),  # minimum number of data needed in a child (leaf)
    'min_child_weight': (1e-5, 1e-3, 'log-uniform'),  # deals with under-fitting
    'subsample': (0.5, 1.0, 'uniform'),  # commonly used range
    'subsample_freq': (1, 10),  # frequency for bagging
    'colsample_bytree': (0.5, 1.0, 'uniform'),  # fraction of features that can be selected for each tree
    'reg_alpha': (1e-9, 10.0, 'log-uniform'),  # L1 regularization term
    'reg_lambda': (1e-9, 10.0, 'log-uniform'),  # L2 regularization term
    'n_estimators': (50, 1000),  # number of boosted trees to fit
}

class LightGBMRegressor(BaseRegressorModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                 test_size=0.3, already_splitted_data=None,  scoring='r2', *args, **kwargs):
        
        super().__init__(train_df=train_df, target_column=target_column, split_column=split_column, create_encoding_rules=create_encoding_rules, 
                         apply_encoding_rules=apply_encoding_rules, test_size=test_size, already_splitted_data=already_splitted_data,
                         scoring=scoring, *args, **kwargs)
        
        self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        self.X_test = DataPreprocessing().set_not_numeric_as_categorial(self.X_test)
        self.estimator = LGBMRegressor( *args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
  