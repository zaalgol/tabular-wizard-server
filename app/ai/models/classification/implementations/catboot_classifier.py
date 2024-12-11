import os
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel


DEFAULT_PARAMS = {
    # 'learning_rate': [0.01, 0.05, 0.1],
    # 'depth': [4, 6, 8, 10],
    # 'l2_leaf_reg': [1, 3, 5, 7, 9],
    # 'iterations': [100, 500, 1000]

    # https://www.kaggle.com/code/lucamassaron/tutorial-bayesian-optimization-with-catboost
    'iterations': [150, 300, 500, 700, 1000],
    'depth': (2, 12, 'int'),
    'learning_rate': (0.01, 0.1, 'log-uniform'),
    'random_strength': (1e-9, 10, 'log-uniform'), # randomness for scoring splits
    'bagging_temperature': (0.0, 1.0), # settings of the Bayesian bootstrap
    'l2_leaf_reg': (2, 20, 'int-log-uniform'), # L2 regularization
}



class CatboostClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, 
                 verbose=False, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        # self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = CatBoostClassifier(verbose=verbose, *args, **kwargs)

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