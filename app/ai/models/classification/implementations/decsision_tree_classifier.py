from sklearn.tree import DecisionTreeClassifier
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

DEFAULT_PARAMS = {
    'criterion': ['gini', 'entropy'],  # The function to measure the quality of a split
    'max_depth': (1, 30, 'int'),  # Maximum depth of the tree
    'min_samples_split': (2, 100, 'int'),  # Minimum number of samples required to split an internal node
    'max_features': ['sqrt', 'log2', None],  # The number of features to consider when looking for the best split
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
