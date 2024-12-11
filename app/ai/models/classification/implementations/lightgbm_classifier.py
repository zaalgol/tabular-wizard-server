# https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier
# https://www.youtube.com/watch?v=AFtjWuwqpSQ

import os
from lightgbm import LGBMClassifier, plot_tree
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt

DEFAULT_PARAMS = {
    'class_weight': ['balanced', None],  # Keep as is, categorical.
    'boosting_type': ['gbdt', 'dart'],  # Keep as is, categorical.
    'num_leaves': (3, 150, 'int'),  # Convert to uniform distribution, specifying as integer is implied.
    'learning_rate': (0.01, 0.1, 'log-uniform'),  # Use log-uniform to explore more granularly at lower values.
    'colsample_bytree': (0.6, 1, 'uniform'),  # Convert to uniform distribution.
    "max_depth": (5, 100, 'int'),  # Keep as uniform, but ensuring integer values are sampled.
    'lambda_l1': (1e-9, 100, 'log-uniform'),  # Keep as log-uniform for fine-grained exploration of regularization.
    'lambda_l2': (1e-9, 100, 'log-uniform')  # Keep as log-uniform for fine-grained exploration of regularization.
}

class LightgbmClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        objective = 'multiclass' if train_df[target_column].nunique() > 2 else 'binary'
        # self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        # self.X_test = DataPreprocessing().set_not_numeric_as_categorial(self.X_test)
        # self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LGBMClassifier(objective=objective, verbosity=-1, **kwargs)

    @property
    def default_params(self):
        default_params = DEFAULT_PARAMS
        return default_params
    
    @property
    def default_values(self):
        return {
            'boosting_type':'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'min_child_weight': 1e-3,
            'subsample': 1.0,
            'subsample_freq': 0,
            'colsample_bytree': 1.0,
            'reg_alpha': 1e-9,
            'reg_lambda': 1e-9,
            'n_estimators': 100,
        }

    
    def train(self):
        # return super().train(eval_metric=self.scoring)
        return super().train()
    

    def save_tree_diagram(self, tree_index=0, model_folder='', filename='tree_diagram.png'):
        plot_tree(self.search.best_estimator_, tree_index=tree_index, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
        plt.savefig(os.path.join(model_folder, filename))
        plt.close()

