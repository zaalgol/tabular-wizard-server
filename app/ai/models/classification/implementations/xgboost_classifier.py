import os
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Real, Categorical, Integer
from app.ai.data_preprocessing import DataPreprocessing 

DEFAULT_PARAMS = {
'learning_rate': Real(0.01, 0.5, prior='log-uniform'),  # Log-uniform is more suitable for learning rates
    'n_estimators': Integer(50, 300),  # Number of trees, integer within a range
    'max_depth': Integer(3, 10, prior='log-uniform'),  # Maximum depth of each tree
    'subsample': Real(0.6, 1.0, prior='uniform'),  # Subsample ratio of the training instances
    'colsample_bytree': Real(0.6, 1.0, prior='uniform'),  # Subsample ratio of columns when constructing each tree
    'colsample_bylevel': Real(0.6, 1.0, prior='uniform'),  # Subsample ratio of columns for each level
    'min_child_weight': Integer(1, 10),  # Minimum sum of instance weight needed in a child
    'gamma': Real(0.0, 0.5, prior='uniform'),  # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': Real(1e-5, 100, prior='log-uniform'),  # L1 regularization term on weights
    'reg_lambda': Real(1e-5, 100, prior='log-uniform'),  # L2 regularization term on weights
    'scale_pos_weight': Real(1, 1000, prior='log-uniform')  # Balancing of positive and negative weights
}

class XgboostClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        
        self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = XGBClassifier(enable_categorical=True, *args, **kwargs)


    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def save_tree_diagram(self, tree_index=0, model_folder='', filename='tree_diagram.png'):
        plot_tree(self.search.best_estimator_, num_trees=tree_index, rankdir='LR')
        fig = plt.gcf()
        fig.set_size_inches(30, 15)
        plt.savefig(os.path.join(model_folder, filename), bbox_inches='tight')
        plt.close()

