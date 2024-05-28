from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from xgboost import plot_importance
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'max_depth': (2, 13, 'int'),
    'learning_rate': (0.01, 0.3, "log-uniform"),
    'subsample': (0.5, 1.0, "uniform"),
    "gamma": (1e-9, 0.5, "log-uniform"),
    'colsample_bytree': (0.5, 1.0, "uniform"),
    'colsample_bylevel': (0.5, 1.0, "uniform"),
    'n_estimators': (100, 1000, 'int'),
    'alpha': (0, 1),
    'lambda': (0, 1),
    'min_child_weight': (1, 10)
}

class XgboostRegressor(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)

        self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = XGBRegressor(enable_categorical=True, *args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
