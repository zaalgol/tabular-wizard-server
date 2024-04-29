from abc import abstractmethod
import os
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from src.models.base_model import BaseModel
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler


class BaseClassfierModel(BaseModel):
        def __init__(self, train_df, target_column, split_column=None, test_size=0.2, scoring='accuracy', sampling_strategy='conditionalOversampling',
                      create_encoding_rules=False, apply_encoding_rules=False, create_transformations=False, apply_transformations=False, *args, **kwargs):
            super().__init__(train_df, target_column, scoring, split_column, 
                             create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                             create_transformations=create_transformations, apply_transformations=apply_transformations, test_size=test_size, *args, **kwargs)
            
            if sampling_strategy == 'conditionalOversampling':
                self.apply_conditional_smote()
            elif sampling_strategy == 'oversampling':
                self.apply_smote()

        def tune_hyper_parameters(self, params=None, kfold=5, n_iter=50):
            if params is None:
                params = self.default_params
            Kfold = KFold(n_splits=kfold)  
            
            self.search = BayesSearchCV(estimator=self.estimator,
                                        search_spaces=params,
                                        scoring=self.scoring,
                                        n_iter=n_iter,
                                        n_jobs=1, 
                                        n_points=3,
                                        cv=Kfold,
                                        verbose=0,
                                        random_state=0)
            
        def train(self):
            if self.search: # with hyperparameter tuining
                result = self.search.fit(self.X_train, self.y_train, callback=self.callbacks)
                print("Best Cross-Validation parameters:", self.search.best_params_)
                print("Best Cross-Validation score:", self.search.best_score_)
            else: # without hyperparameter tuining
                return self.estimator.fit(self.X_train, self.y_train)
            return result
        
        def apply_conditional_smote(self):
            class_counts = self.y_train.value_counts()

            smallest_class = class_counts.min()
            largest_class = class_counts.max()
            ratio = smallest_class / largest_class

            # Define a threshold below which we consider the dataset imbalanced
            # This threshold can be adjusted based on specific needs
            imbalance_threshold = 0.5  # Example threshold

            # If the ratio is below the threshold, apply SMOTE
            if ratio < imbalance_threshold:
                self.apply_smote()
            else:
                print("The dataset is considered balanced. Skipping SMOTE.")
                
        def apply_smote(self):
            random_pver_sampler = RandomOverSampler(random_state=0)
            self.X_train, self.y_train = random_pver_sampler.fit_resample(self.X_train, self.y_train)
        
        def save_feature_importances(self, model_folder='', filename='feature_importances.png'):
            # Default implementation, to be overridden in derived classes
            feature_importances = self.search.best_estimator_.feature_importances_
            feature_names = self.X_train.columns
            plt.figure(figsize=(12, 6))
            plt.barh(feature_names, feature_importances)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.savefig(os.path.join(model_folder, filename))
            plt.close()

