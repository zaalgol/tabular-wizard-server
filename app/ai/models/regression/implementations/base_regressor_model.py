# app\ai\models\regression\implementations\base_regressor_model.py

from sklearn.model_selection import KFold, cross_val_score
import optuna
from sklearn.base import clone
from app.ai.models.base_model import BaseModel


class BaseRegressorModel(BaseModel):
    def __init__(self, train_df, target_column, scoring='r2', *args, **kwargs):
        super().__init__(train_df, target_column, scoring, *args, **kwargs)

    def tune_hyper_parameters(self, params=None, kfold=5, n_iter=500, timeout=45*60, *args, **kwargs):
        if params is None:
            params = self.default_params
        # Configure KFold with shuffling for better generalization
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

        def objective(trial):
            param_grid = {}
            for k, v in params.items():
                if isinstance(v, list):
                    param_grid[k] = trial.suggest_categorical(k, v)
                elif isinstance(v, tuple) and len(v) == 3:
                    if v[2] == 'log-uniform':
                        param_grid[k] = trial.suggest_float(k, v[0], v[1], log=True)
                    elif v[2] == 'uniform':
                        param_grid[k] = trial.suggest_float(k, v[0], v[1])
                    elif v[2] == 'int':
                        param_grid[k] = trial.suggest_int(k, v[0], v[1])
                    elif v[2] == 'int-log-uniform':
                        # Convert float suggestion to integer if needed
                        param_grid[k] = int(trial.suggest_float(k, v[0], v[1], log=True))
                    else:
                        raise ValueError(f"Unsupported distribution type for parameter {k}: {v[2]}")
                elif isinstance(v, tuple) and len(v) == 2:
                    # Assuming uniform float if only two elements are present
                    param_grid[k] = trial.suggest_float(k, v[0], v[1])
                else:
                    raise ValueError(f"Unsupported parameter format for {k}: {v}")

            # Clone the estimator to ensure independence between trials
            estimator = clone(self.estimator)
            estimator.set_params(**param_grid)

            # Perform cross-validation
            try:
                cv_results = cross_val_score(estimator, self.X_train, self.y_train, cv=kf, scoring=self.scoring, n_jobs=-1)
                mean_score = cv_results.mean()
                trial.report(mean_score, step=0)

                # Handle pruning based on intermediate results if desired
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return mean_score
            except Exception as e:
                # Optionally, handle exceptions or log them
                print(f"Exception during trial {trial.number}: {e}")
                return float('-inf')  # Assign a very low score to pruned trials

        # Create and optimize the study
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=n_iter, timeout=timeout)

    def train(self):
        if self.study:
            best_params = self.study.best_params
            self.estimator.set_params(**best_params)
            result = self.estimator.fit(self.X_train, self.y_train)
            print("Best parameters:", best_params)
            print("Best cross-validation score:", self.study.best_value)
        else:
            result = self.estimator.fit(self.X_train, self.y_train)
        return result

    @property
    def unnecessary_parameters(self):
        return [
            'scoring', 'split_column', 'create_encoding_rules', 'apply_encoding_rules',
            'create_transformations', 'apply_transformations', 'test_size',
            'already_splitted_data'
        ]
