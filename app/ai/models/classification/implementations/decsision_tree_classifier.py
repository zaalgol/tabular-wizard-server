from sklearn.tree import DecisionTreeClassifier
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Categorical, Integer

DEFAULT_PARAMS = {
    'criterion': Categorical(['gini', 'entropy']),  # The function to measure the quality of a split
    'max_depth': Integer(1, 30),  # Maximum depth of the tree
    'min_samples_split': Integer(2, 20),  # Minimum number of samples required to split an internal node
    'max_features': Categorical(['auto', 'sqrt', 'log2', None]),  # The number of features to consider when looking for the best split
}

class DecisionTreeClassifierWrapper(BaseClassfierModel):
    def __init__(self, train_df, target_column, 
                  *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = DecisionTreeClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
