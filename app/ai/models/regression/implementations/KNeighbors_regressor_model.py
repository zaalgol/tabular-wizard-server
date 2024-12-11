from sklearn.neighbors import KNeighborsRegressor
from app.ai.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS = {
    'n_neighbors': (1, 30, "int"),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': (10, 100, "int"),
    'p': (1, 2, "int")
}

class KNeighborsRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        # self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = KNeighborsRegressor(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
