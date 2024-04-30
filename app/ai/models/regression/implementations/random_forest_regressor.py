from sklearn.ensemble import RandomForestRegressor
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS_RF = {
    'n_estimators': (100, 1000),
    'max_features': (0.1, 1.0, "uniform"),
    'max_depth': (3, 30),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'bootstrap': [True, False]
}

class RandomForestRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = RandomForestRegressor(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS_RF
