from sklearn.neighbors import KNeighborsClassifier
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Categorical, Integer


DEFAULT_PARAMS = {
    'n_neighbors': Integer(1, 30),  # Number of neighbors
    'weights': Categorical(['uniform', 'distance']),  # Weight type
    'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),  # Algorithm used to compute the nearest neighbors
    'leaf_size': Integer(20, 50),  # Leaf size passed to BallTree or KDTree
}

class KnnClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False, 
                 create_transformations=False, apply_transformations=False, 
                 test_size=0.3, already_splitted_data=None, sampling_strategy='conditionalOversampling', *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size, 
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                         create_transformations=create_transformations, apply_transformations=apply_transformations, 
                         already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)

        self.estimator = KNeighborsClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
