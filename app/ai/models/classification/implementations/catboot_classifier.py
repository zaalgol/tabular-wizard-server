import os
import matplotlib.pyplot as plt
from skopt.space import Real, Categorical, Integer

from catboost import CatBoostClassifier
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel


DEFAULT_PARAMS = {
    # 'learning_rate': [0.01, 0.05, 0.1],
    # 'depth': [4, 6, 8, 10],
    # 'l2_leaf_reg': [1, 3, 5, 7, 9],
    # 'iterations': [100, 500, 1000]

    # https://www.kaggle.com/code/lucamassaron/tutorial-bayesian-optimization-with-catboost
    'iterations': Integer(20, 150, 'log-uniform'),
    'depth': Integer(2, 12),
    'learning_rate': Real(0.01, 0.1, 'log-uniform'),
    'random_strength': Real(1e-9, 10, 'log-uniform'), # randomness for scoring splits
    'bagging_temperature': Real(0.0, 1.0), # settings of the Bayesian bootstrap
    'l2_leaf_reg': Integer(2, 20, 'log-uniform'), # L2 regularization
}



class CatboostClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, split_column=None, test_size=0.3, 
                 create_encoding_rules=False, apply_encoding_rules=False, already_splitted_data=None,  sampling_strategy='conditionalOversampling',
                 verbose=False, *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,
                          create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                          already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)
        self.estimator = CatBoostClassifier(task_type = 'GPU', devices='0', verbose=verbose, *args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def save_feature_importances(self, model_folder='', filename='catboost_feature_importances.png'):
        feature_importances = self.search.best_estimator_.get_feature_importance()
        feature_names = self.X_train.columns
        plt.figure(figsize=(12, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.savefig(os.path.join(model_folder, filename),)
        plt.close()