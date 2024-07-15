from abc import abstractmethod
import os
from sklearn.model_selection import KFold, cross_val_score
import optuna
from app.ai.models.base_model import BaseModel
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.models.classification.evaluate import Evaluate
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score, log_loss, precision_score, recall_score, roc_auc_score, f1_score


class BaseClassfierModel(BaseModel):
    def __init__(self, train_df, target_column, split_column=None, scoring='accuracy', sampling_strategy='conditionalOversampling',
                 create_encoding_rules=False, apply_encoding_rules=False, create_transformations=False, apply_transformations=False, *args, **kwargs):
        super().__init__(train_df, target_column, scoring, split_column,
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         create_transformations=create_transformations, apply_transformations=apply_transformations, *args, **kwargs)
        if sampling_strategy == 'conditionalOversampling':
            self.apply_conditional_oversampling()
        elif sampling_strategy == 'oversampling':
            self.apply_oversampling()
        self.is_multi_class = DataPreprocessing().get_class_num(self.y_train) > 2
        self.scoring = Evaluate().get_scoring_metric(scoring, self.is_multi_class)

    def tune_hyper_parameters(self, params=None, kfold=5, n_iter=50, timeout=45*60):
        if params is None:
            params = self.default_params
        kfold = KFold(n_splits=kfold)

        def objective(trial):
            param_grid = {}
            for k, v in params.items():
                if isinstance(v, tuple) and len(v) == 3 and v[2] == 'log-uniform':
                    param_grid[k] = trial.suggest_float(k, v[0], v[1], log=True)
                elif isinstance(v, tuple) and len(v) == 3 and v[2] == 'uniform':
                    param_grid[k] = trial.suggest_float(k, v[0], v[1])
                elif isinstance(v, tuple) and len(v) == 3 and v[2] == 'int':
                    param_grid[k] = trial.suggest_int(k, v[0], v[1])
                elif isinstance(v, tuple) and len(v) == 3 and v[2] == 'int-log-uniform':
                    param_grid[k] = int(trial.suggest_float(k, v[0], v[1], log=True))
                elif isinstance(v, tuple) and len(v) == 2:
                    param_grid[k] = trial.suggest_float(k, v[0], v[1])
                elif isinstance(v, list):
                    param_grid[k] = trial.suggest_categorical(k, v)
                else:
                    raise ValueError(f"Unsupported parameter format for {k}: {v}")
                
            cv_results = cross_val_score(self.estimator, self.X_train, self.y_train, cv=kfold, scoring=self.scoring)
            return cv_results.mean()

        self.study = optuna.create_study(direction="maximize") #if self.scoring in ["accuracy", "roc_auc", "f1_macro"] else "minimize")
        # self.study.enqueue_trial(self.default_values)
        self.study.optimize(objective, n_trials=n_iter, timeout=timeout)

    def train(self, *args, **kwargs):
        if self.study:  # with hyperparameter tuning
            best_params = self.study.best_params
            self.estimator.set_params(**best_params)
            result = self.estimator.fit(self.X_train, self.y_train, *args, **kwargs)
            print("Best Cross-Validation parameters:", best_params)
            print("Best Cross-Validation score:", self.study.best_value)
        else:  # without hyperparameter tuning
            result = self.estimator.fit(self.X_train, self.y_train, *args, **kwargs)
        return result

    def apply_conditional_oversampling(self):
        class_counts = self.y_train.value_counts()

        smallest_class = class_counts.min()
        largest_class = class_counts.max()
        ratio = smallest_class / largest_class

        # Define a threshold below which we consider the dataset imbalanced
        # This threshold can be adjusted based on specific needs
        imbalance_threshold = 0.5  # Example threshold

        # If the ratio is below the threshold, apply oversampling
        if ratio < imbalance_threshold:
            self.apply_oversampling()
        else:
            print("The dataset is considered balanced. Skipping oversampling.")
                
    def apply_oversampling(self):
        class_counts = self.y_train.value_counts()
        max_size = class_counts.max()

        X_train_resampled = []
        y_train_resampled = []

        for class_index, count in class_counts.items():
            df_class_indices = self.y_train[self.y_train == class_index].index
            df_class = self.X_train.loc[df_class_indices]
            if count < max_size:
                df_class_over = resample(df_class, 
                                         replace=True,  # sample with replacement
                                         n_samples=max_size,  # match number in majority class
                                         random_state=42)  # reproducible results
                y_class_over = resample(self.y_train.loc[df_class_indices], 
                                        replace=True, 
                                        n_samples=max_size, 
                                        random_state=42)
                X_train_resampled.append(df_class_over)
                y_train_resampled.append(y_class_over)
            else:
                X_train_resampled.append(df_class)
                y_train_resampled.append(self.y_train.loc[df_class_indices])
        
        self.X_train = pd.concat(X_train_resampled)
        self.y_train = pd.concat(y_train_resampled)

    @property
    def unnecessary_parameters(self):
        return ['scoring', 'split_column', 'create_encoding_rules', 'apply_encoding_rules', 'create_transformations', 'apply_transformations', 'test_size',
                'already_splitted_data', 'sampling_strategy']
    
    def save_feature_importances(self, model_folder='', filename='feature_importances.png'):
        # Default implementation, to be overridden in derived classes
        feature_importances = self.estimator.feature_importances_
        feature_names = self.X_train.columns
        plt.figure(figsize=(12, 6))
        plt.barh(feature_names, feature_importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.savefig(os.path.join(model_folder, filename))
        plt.close()
