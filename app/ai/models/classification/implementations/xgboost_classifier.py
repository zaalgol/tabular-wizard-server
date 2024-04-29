import os
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Real, Categorical, Integer
from src.data_preprocessing import DataPreprocessing 

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
    def __init__(self, train_df, target_column, split_column=None,
                 create_encoding_rules=False, apply_encoding_rules=False,
                 test_size=0.3, already_splitted_data=None,  sampling_strategy='conditionalOversampling', *args, **kwargs):
        
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)
        
        self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
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

    # def plot (self, result):
    #     from xgboost import plot_importance
    #     import matplotlib.pyplot as plt
    #     plt.style.use ('fivethirtyeight')
    #     plt.rcParams.update ({'font.size': 16})

    #     fig, ax = plt.subplots (figsize=(12, 6))
    #     plot_importance (result.best_estimator_, max_num_features=8, ax=ax)
    #     plt.show()

    #     # xgb.plot_tree (result.best_estimator_, num_trees=2)
    #     # fig = matplotlib.pyplot.gcf ()
    #     # fig.set_size_inches (150, 100)
    #     # fig.savefig ('tree.png')

    #     # plot_tree (result.best_estimator_)

    # def predict_test (self):
    #     self.test_predictions = self.search.predict (self.X_test)

    # def predict_train (self):
    #     self.train_predictions = self.search.predict (self.X_train)

    # def predict_true (self):
    #     self.true_predictions = self.search.predict (self.X_true)

    # def evaluate_predictions (self):
    #     cm = confusion_matrix (self.y_test, self.test_predictions)
    #     print (classification_report (self.y_test, self.test_predictions))
    #     print (f"confusion_matrix of test is {cm}")

    #     cm = confusion_matrix (self.y_train, self.train_predictions)
    #     print (classification_report (self.y_train, self.train_predictions))
    #     print (f"confusion_matrix of train is {cm}")
