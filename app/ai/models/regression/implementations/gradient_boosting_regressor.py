from sklearn.ensemble import GradientBoostingRegressor
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'n_estimators': (50, 500, "int"),
    'learning_rate': (0.01, 0.3, "uniform"),
    'max_depth': (1, 10, "int"),
    'subsample': (0.5, 1.0, "uniform"),
    'min_samples_split': (2, 20, "int"),
    'min_samples_leaf': (1, 20, "int"),
    'max_features': ['auto', 'sqrt', 'log2', None]
}

class GBRegressor(BaseRegressorModel):
    def __init__(self, target_column, scoring, *args, **kwargs):
        super().__init__(target_column, scoring, *args, **kwargs)
        # self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = GradientBoostingRegressor(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
