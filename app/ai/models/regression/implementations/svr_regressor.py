from sklearn.svm import SVR
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS_SVR = {
    'C': (0.1, 1000, "log-uniform"),
    'epsilon': (0.01, 1.0, "uniform"),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': (1, 5)  # Only relevant for poly kernel
}

class SVRRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = SVR(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS_SVR
