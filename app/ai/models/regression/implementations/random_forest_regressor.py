from sklearn.ensemble import RandomForestRegressor
from src.data_preprocessing import DataPreprocessing
from src.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS_RF = {
    'n_estimators': (100, 1000),
    'max_features': (0.1, 1.0, "uniform"),
    'max_depth': (3, 30),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'bootstrap': [True, False]
}

class RandomForestRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, split_column=None,
                 create_encoding_rules=False, apply_encoding_rules=False,
                 test_size=0.3, already_splitted_data=None, scoring='r2', *args, **kwargs):
        
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         already_splitted_data=already_splitted_data, scoring=scoring, *args, **kwargs)
        
        self.estimator = RandomForestRegressor(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS_RF
