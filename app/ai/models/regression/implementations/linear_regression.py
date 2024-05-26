from sklearn.linear_model import LinearRegression
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False]
}

class LinearRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LinearRegression(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
