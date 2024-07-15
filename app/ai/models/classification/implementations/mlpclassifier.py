import math
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt

DEFAULT_PARAMS = {
    # 'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
    'hidden_layer_sizes': [ (100,), (20, 10) ],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.0001, 0.05, 0.1, 0.5, 1, 2, 3, 4],
    'learning_rate_init': [0.001, 0.01, 0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

class MLPNetClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, hidden_layer_sizes=None, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
         # Choose the solver based on the number of rows in the dataset
        # if len(train_df) <= 1000 and 'solver' not in kwargs:
        #      kwargs['solver'] = 'lbfgs'
        if not hidden_layer_sizes:
            first_layer_size=max(len(self.X_train.columns), 2)
            second_layer_size=max(int(first_layer_size /2), 2)
            hidden_layer_sizes=(first_layer_size, second_layer_size)
            
            
        self.estimator = MLPClassifier(max_iter=500, hidden_layer_sizes=hidden_layer_sizes, *args, **kwargs)

    def train(self):
            if self.search: # with hyperparameter tuining
                result = self.search.fit(self.X_train, self.y_train)
                print("Best Cross-Validation parameters:", self.search.best_params_)
                print("Best Cross-Validation score:", self.search.best_score_)
            else:
                result = self.estimator.fit(self.X_train, self.y_train)
                # print("Best accuracy:", self.estimator.best_score_)
            return result
        

    def tune_hyper_parameters(self, params=None, scoring='accuracy', kfold=10, n_iter=150):
            if params is None:
                params = self.default_params
            Kfold = KFold(n_splits=kfold)  
            
            self.search = GridSearchCV(estimator=self.estimator,
                                        param_grid=params,
                                        scoring=scoring,
                                        # n_iter=n_iter,
                                        n_jobs=1, 
                                        cv=Kfold,
                                        verbose=0)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def visualize_weights(self,  model_folder='', filename='mlp_weights_visualization.png'):
        # layers = len(self.search.best_estimator_.coefs_)
        # fig, axes = plt.subplots(nrows=1, ncols=layers, figsize=(20, 5))
        # for i in range(layers):
        #     ax = axes[i]
        #     ax.matshow(self.search.best_estimator_.coefs_[i], cmap='viridis')
        #     ax.set_title(f'Layer {i+1}')
        #     ax.set_xlabel('Neurons in Layer')
        #     ax.set_ylabel('Input Features')
        # plt.tight_layout()
        # plt.savefig(os.path.join(model_folder, filename))
        # plt.close(fig)
        layers = len(self.search.best_estimator_.coefs_)
    # Determine the appropriate figure size dynamically, for example:
        fig_width = max(12, layers * 4)  # Adjust as necessary
        fig_height = max(4, layers)  # Adjust as necessary
        fig, axes = plt.subplots(nrows=1, ncols=layers, figsize=(fig_width, fig_height))
        for i in range(layers):
            ax = axes[i] if layers > 1 else axes
            im = ax.matshow(self.search.best_estimator_.coefs_[i], cmap='viridis', interpolation='none')
            ax.set_title(f'Layer {i+1}')
            ax.set_xlabel('Neurons in Layer')
            ax.set_ylabel('Input Features')
            # Optionally add a colorbar to each subplot
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, filename), dpi=300)  # Increase DPI for higher resolution
        plt.close(fig)
