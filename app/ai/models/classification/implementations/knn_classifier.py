from sklearn.neighbors import KNeighborsClassifier
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Categorical, Integer


DEFAULT_PARAMS = {
    'n_neighbors': Integer(1, 30),  # Number of neighbors
    'weights': Categorical(['uniform', 'distance']),  # Weight type
    'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),  # Algorithm used to compute the nearest neighbors
    'leaf_size': Integer(20, 50),  # Leaf size passed to BallTree or KDTree
}

class KnnClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, 
                  *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = KNeighborsClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
