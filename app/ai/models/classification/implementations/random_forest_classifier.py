from sklearn.ensemble import RandomForestClassifier
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from skopt.space import Real, Categorical, Integer
import os

DEFAULT_PARAMS = {
    'n_estimators': Integer(50, 300),  # Number of trees in the forest
    'max_depth': Categorical([3, 4, 5, 6, 7, 8, 9, None]),  # Maximum depth of each tree
    'min_samples_split': Categorical([2, 5, 10]),  # Minimum number of samples required to split an internal node
    'min_samples_leaf': Categorical([1, 2, 4]),  # Minimum number of samples required to be at a leaf node
    'bootstrap': Categorical([True, False]),  # Method for sampling data points (with or without replacement)
    'criterion': Categorical(['gini', 'entropy']),  # The function to measure the quality of a split
    'max_features': Integer(1, 19)  # The number of features to consider when looking for the best split
}

class RandomForestClassifierCustom(BaseClassfierModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False, test_size=0.3,
                 already_splitted_data=None, sampling_strategy='conditionalOversampling', *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,  create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)
        self.estimator = RandomForestClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS

    # def train(self):
    #     result = self.search.fit(self.X_train, self.y_train.values.ravel())  # using values.ravel() to get a 1-D array
    #     print("Best parameters:", self.search.best_params_)
    #     print("Best accuracy:", self.search.best_score_)

    #     return result
    
    def save_tree_diagram(self, tree_index=0, model_folder='', filename='random_forest_tree_diagram.png', dpi=300):
        plt.figure(figsize=(20, 10))
        plot_tree(self.estimator.estimators_[tree_index], filled=True, feature_names=self.X_train.columns, rounded=True, class_names=True)
        plt.savefig(os.path.join(model_folder, filename), format='png', dpi=dpi)
        plt.close()
