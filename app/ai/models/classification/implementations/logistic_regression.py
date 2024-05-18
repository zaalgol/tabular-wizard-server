from sklearn.linear_model import LogisticRegression
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Real, Categorical, Integer

DEFAULT_PARAMS = [
    {
        'solver': Categorical(['newton-cg', 'lbfgs', 'sag']),  # Solvers that support only l2 or None
        'penalty': Categorical(['l2', None]),
        'class_weight': Categorical(['balanced', None]),
        'C': Real(0.0001, 10, prior='log-uniform'),
        'fit_intercept': Categorical([True, False]),
        'max_iter': Integer(100, 1000),
        'tol': Real(1e-6, 1e-2, prior='log-uniform')
    },
    {
        'solver': Categorical(['liblinear']),  # Solver that supports l1, l2, or None
        'penalty': Categorical(['l1', 'l2']),
        'class_weight': Categorical(['balanced', None]),
        'C': Real(0.0001, 10, prior='log-uniform'),
        'fit_intercept': Categorical([True, False]),
        'max_iter': Integer(100, 1000),
        'tol': Real(1e-6, 1e-2, prior='log-uniform')
    },
    {
        'solver': Categorical(['saga']),  # Solver that supports l1, l2, elasticnet, or None
        'penalty': Categorical(['l1', 'l2', None]),
        'class_weight': Categorical(['balanced', None]),
        'C': Real(0.0001, 10, prior='log-uniform'),
        'fit_intercept': Categorical([True, False]),
        'max_iter': Integer(100, 1000),
        'tol': Real(1e-6, 1e-2, prior='log-uniform')  # Only relevant for 'elasticnet'
    },
    {
        'solver': Categorical(['saga']),  # Solver that supports l1, l2, elasticnet, or None
        'penalty': Categorical(['elasticnet']),
        'class_weight': Categorical(['balanced', None]),
        'C': Real(0.0001, 10, prior='log-uniform'),
        'fit_intercept': Categorical([True, False]),
        'max_iter': Integer(100, 1000),
        'tol': Real(1e-6, 1e-2, prior='log-uniform'),
        'l1_ratio': Real(0, 1, prior='uniform')  # Only relevant for 'elasticnet'
    }
]


class LRegression(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LogisticRegression(*args, **kwargs)



    @property
    def default_params(self):
        return DEFAULT_PARAMS
