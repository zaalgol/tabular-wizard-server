import os
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from app.ai.data_preprocessing import DataPreprocessing 

DEFAULT_PARAMS = {
'learning_rate': (0.01, 0.5, 'log-uniform'),  # Log-uniform is more suitable for learning rates
    'n_estimators': (50, 300, 'int'),  # Number of trees,  within a range
    'max_depth': (3, 10, 'log-uniform'),  # Maximum depth of each tree
    'subsample': (0.6, 1.0, 'uniform'),  # Subsample ratio of the training instances
    'colsample_bytree': (0.6, 1.0, 'uniform'),  # Subsample ratio of columns when constructing each tree
    'colsample_bylevel': (0.6, 1.0, 'uniform'),  # Subsample ratio of columns for each level
    'min_child_weight': (1, 10,'int'),  # Minimum sum of instance weight needed in a child
    'gamma': (0.0, 0.5, 'uniform'),  # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': (1e-5, 100, 'log-uniform'),  # L1 regularization term on weights
    'reg_lambda': (1e-5, 100, 'log-uniform'),  # L2 regularization term on weights
    'scale_pos_weight': (1, 1000, 'log-uniform')  # Balancing of positive and negative weights
}

class XgboostClassifier(BaseClassfierModel):
    def __init__(self, target_column, scoring, *args, **kwargs):
        super().__init__(target_column, scoring, *args, **kwargs)
        
        # self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        self.estimator = XGBClassifier(enable_=True, *args, **kwargs)


    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def save_tree_diagram(self, tree_index=0, model_folder='', filename='tree_diagram.png'):
        plot_tree(self.search.best_estimator_, num_trees=tree_index, rankdir='LR')
        fig = plt.gcf()
        fig.set_size_inches(30, 15)
        plt.savefig(os.path.join(model_folder, filename), bbox_inches='tight')
        plt.close()

