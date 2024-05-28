from sklearn.ensemble import RandomForestClassifier
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

DEFAULT_PARAMS = {
    'n_estimators': (50, 300, 'int'),  # Number of trees in the forest
    'max_depth': [3, 4, 5, 6, 7, 8, 9, None],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False],  # Method for sampling data points (with or without replacement)
    'criterion': ['gini', 'entropy'],  # The function to measure the quality of a split
    'max_features': (1, 19, 'int')  # The number of features to consider when looking for the best split
}

class RandomForestClassifierCustom(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = RandomForestClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def save_tree_diagram(self, tree_index=0, model_folder='', filename='random_forest_tree_diagram.png', dpi=300):
        plt.figure(figsize=(20, 10))
        plot_tree(self.estimator.estimators_[tree_index], filled=True, feature_names=self.X_train.columns, rounded=True, class_names=True)
        plt.savefig(os.path.join(model_folder, filename), format='png', dpi=dpi)
        plt.close()
