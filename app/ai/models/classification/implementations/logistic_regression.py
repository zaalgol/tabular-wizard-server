from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
import optuna

# Define all valid solver-penalty combinations
SOLVER_PENALTY_COMBINATIONS = [
    ('newton-cg', 'l2'),
    ('lbfgs', 'l2'),
    ('sag', 'l2'),
    ('liblinear', 'l1'),
    ('liblinear', 'l2'),
    ('saga', 'l1'),
    ('saga', 'l2'),
    ('saga', 'elasticnet'),
    ('saga', 'none')
]

# Default parameters for Logistic Regression
DEFAULT_PARAMS = {
    'solver_penalty': SOLVER_PENALTY_COMBINATIONS,
    'C': (0.0001, 10, 'log-uniform'),
    'fit_intercept': [True, False],
    'max_iter': (100, 1000, 'int'),
    'tol': (1e-6, 1e-2, 'log-uniform'),
    'class_weight': ['balanced', None],
    'l1_ratio': (0, 1, 'uniform')  # Only used when penalty is 'elasticnet'
}

class LRegression(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LogisticRegression(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS

    def tune_hyper_parameters(self, params=None, kfold=5, n_iter=300, timeout=60*60):
        if params is None:
            params = self.default_params

        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

        def objective(trial):
            solver, penalty = trial.suggest_categorical('solver_penalty', params['solver_penalty'])
            
            param_grid = {
                'solver': solver,
                'penalty': penalty,
                'fit_intercept': trial.suggest_categorical('fit_intercept', params['fit_intercept']),
                'class_weight': trial.suggest_categorical('class_weight', params['class_weight'])
            }

            # Handle parameters with specific distribution types
            c_low, c_high, c_dist = params['C']
            param_grid['C'] = trial.suggest_float('C', c_low, c_high, log=(c_dist == 'log-uniform'))

            max_iter_low, max_iter_high, _ = params['max_iter']
            param_grid['max_iter'] = trial.suggest_int('max_iter', max_iter_low, max_iter_high)

            tol_low, tol_high, tol_dist = params['tol']
            param_grid['tol'] = trial.suggest_float('tol', tol_low, tol_high, log=(tol_dist == 'log-uniform'))

            if penalty == 'elasticnet':
                l1_ratio_low, l1_ratio_high, _ = params['l1_ratio']
                param_grid['l1_ratio'] = trial.suggest_float('l1_ratio', l1_ratio_low, l1_ratio_high)
            
            if penalty == 'none':
                param_grid['penalty'] = None

            estimator = LogisticRegression(**param_grid)
            cv_results = cross_val_score(estimator, self.X_train, self.y_train, cv=kf, scoring=self.scoring)
            return cv_results.mean()

        # Create and optimize the study
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=n_iter, timeout=timeout)

    def train(self, *args, **kwargs):
        if self.study:
            best_params = self.study.best_params
            solver, penalty = best_params.pop('solver_penalty')
            best_params['solver'] = solver
            best_params['penalty'] = None if penalty == 'none' else penalty
            
            if 'l1_ratio' in best_params and best_params['penalty'] != 'elasticnet':
                best_params.pop('l1_ratio')
            
            self.estimator = LogisticRegression(**best_params)
        
        return self.estimator.fit(self.X_train, self.y_train, *args, **kwargs)