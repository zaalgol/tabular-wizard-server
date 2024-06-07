from sklearn.model_selection import KFold, cross_val_score
import optuna
from app.ai.models.base_model import BaseModel

class BaseRegressorModel(BaseModel):
    def __init__(self, train_df, target_column, scoring='r2', *args, **kwargs):
        super().__init__(train_df, target_column, scoring, *args, **kwargs)

    def tune_hyper_parameters(self, params=None, kfold=5, n_iter=50, timeout=45*60, *args, **kwargs):
        if params is None:
            params = self.default_params
        kfold = KFold(n_splits=kfold)

        def objective(trial):
            param_grid = {}
            for k, v in params.items():
                if isinstance(v, list):
                    param_grid[k] = trial.suggest_categorical(k, v)
                elif isinstance(v, tuple) and len(v) == 3 and v[2] == 'log-uniform':
                    param_grid[k] = trial.suggest_float(k, v[0], v[1], log=True)
                elif isinstance(v, tuple) and len(v) == 3 and v[2] == 'uniform':
                    param_grid[k] = trial.suggest_float(k, v[0], v[1])
                elif isinstance(v, tuple) and len(v) == 3 and v[2] == 'int':
                    param_grid[k] = trial.suggest_int(k, v[0], v[1])
                elif isinstance(v, tuple) and len(v) == 2:
                    param_grid[k] = trial.suggest_float(k, v[0], v[1])
                else:
                    raise ValueError(f"Unsupported parameter format for {k}: {v}")

            estimator = self.estimator.set_params(**param_grid)
            cv_results = cross_val_score(estimator, self.X_train, self.y_train, cv=kfold, scoring=self.scoring)
            return cv_results.mean()

        self.study = optuna.create_study(direction="maximize")# if self.scoring in ["neg_root_mean_squared_error", "roc_auc", "f1", "r2"] else "minimize")

        # Add a trial with default parameters
        self.study.enqueue_trial(self.default_values)

        self.study.optimize(objective, n_trials=n_iter, timeout=timeout)

    def train(self):
        if self.study:
            best_params = self.study.best_params
            self.estimator.set_params(**best_params)
            result = self.estimator.fit(self.X_train, self.y_train)
            print("Best parameters:", best_params)
        else:
            result = self.estimator.fit(self.X_train, self.y_train)
        return result

    @property
    def unnecessary_parameters(self):
        return ['scoring', 'split_column', 'create_encoding_rules', 'apply_encoding_rules', 'create_transformations', 'apply_transformations', 'test_size',
                'already_splitted_data']
