from sklearn.linear_model import SGDRegressor
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'alpha': (0.0001, 1000, 'log-uniform'),
    'penalty': ['l2'],
    'max_iter': [100, 1000,2500, 5000],
    'fit_intercept': [True, False]
}

class LinearRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = SGDRegressor(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
