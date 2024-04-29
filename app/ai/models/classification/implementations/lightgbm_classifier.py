# https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier
# https://www.youtube.com/watch?v=AFtjWuwqpSQ

import os
from lightgbm import LGBMClassifier, plot_tree
from src.data_preprocessing import DataPreprocessing
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt
from skopt.space import Real, Categorical, Integer

DEFAULT_PARAMS_OLD = {
    'class_weight': ['balanced', None],  # Keep as is, categorical.
    'boosting_type': ['gbdt', 'dart'],  # Keep as is, categorical.
    'num_leaves': (3, 150, 'uniform'),  # Convert to uniform distribution, specifying as integer is implied.
    'learning_rate': (0.01, 0.1, 'log-uniform'),  # Use log-uniform to explore more granularly at lower values.
    'subsample_for_bin': (20000, 150000, 'uniform'),  # Convert to uniform distribution.
    'min_child_samples': (20, 500, 'uniform'),  # Convert to uniform distribution.
    'colsample_bytree': (0.6, 1, 'uniform'),  # Convert to uniform distribution.
    "max_depth": (5, 100, 'uniform'),  # Keep as uniform, but ensuring integer values are sampled.
    'lambda_l1': (1e-9, 100, 'log-uniform'),  # Keep as log-uniform for fine-grained exploration of regularization.
    'lambda_l2': (1e-9, 100, 'log-uniform')  # Keep as log-uniform for fine-grained exploration of regularization.
}

DEFAULT_PARAMS = {
    'learning_rate': Real(0.01, 0.1, 'log-uniform'),     # Boosting learning rate
    'n_estimators': Integer(30, 5000),                   # Number of boosted trees to fit
    # 'n_estimators': Integer(30, 5000),                   # Number of boosted trees to fit
    'num_leaves': Integer(2, 512),                       # Maximum tree leaves for base learners
    'max_depth': Integer(-1, 256),                       # Maximum tree depth for base learners, <=0 means no limit
    'min_child_samples': Integer(1, 256),                # Minimal number of data in one leaf
    'max_bin': Integer(100, 1000),                       # Max number of bins that feature values will be bucketed
    'subsample': Real(0.01, 1.0, 'uniform'),             # Subsample ratio of the training instance
    'subsample_freq': Integer(0, 10),                    # Frequency of subsample, <=0 means no enable
    'colsample_bytree': Real(0.01, 1.0, 'uniform'),      # Subsample ratio of columns when constructing each tree
    'min_child_weight': Real(0.01, 10.0, 'uniform'),     # Minimum sum of instance weight (hessian) needed in a child (leaf)
    'reg_lambda': Real(1e-9, 100.0, 'log-uniform'),      # L2 regularization
    'reg_alpha': Real(1e-9, 100.0, 'log-uniform'),       # L1 regularization
}

class LightgbmClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                 test_size=0.3, scoring='accuracy', sampling_strategy='conditionalOversampling', already_splitted_data=None, *args, **kwargs):
        super().__init__(train_df=train_df, target_column=target_column, split_column=split_column, create_encoding_rules=create_encoding_rules, 
                         apply_encoding_rules=apply_encoding_rules, test_size=test_size, scoring=scoring,
                         already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)
        objective = 'multiclass' if train_df[target_column].nunique() > 2 else 'binary'
        self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        self.X_test = DataPreprocessing().set_not_numeric_as_categorial(self.X_test)
        self.estimator = LGBMClassifier(objective=objective, *args, **kwargs)

    @property
    def default_params(self):
        default_params = DEFAULT_PARAMS_OLD
        return default_params
    

    def save_tree_diagram(self, tree_index=0, model_folder='', filename='tree_diagram.png'):
        plot_tree(self.search.best_estimator_, tree_index=tree_index, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
        plt.savefig(os.path.join(model_folder, filename))
        plt.close()

