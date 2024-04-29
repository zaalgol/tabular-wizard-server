from sklearn.linear_model import LogisticRegression
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer


# DEFAULT_PARAMS= {
#     'class_weight': ['balanced', None],
#     'penalty': ['l1', 'l2', 'elasticnet', None],  # Penalty term, categorical. Choose from L1, L2, ElasticNet or none.
#     'C': (1e-4, 10, 'log-uniform'),  # Inverse of regularization strength; smaller values specify stronger regularization.
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in the optimization problem, categorical.
#     'class_weight': ['balanced', None],  # Weights associated with classes, categorical.
#     'l1_ratio': (0, 1, 'uniform'),  # The Elastic-Net mixing parameter, only used if penalty is 'elasticnet'.
#     'fit_intercept': [True, False],  # Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
#     'max_iter': (100, 1000, 'uniform'),  # Maximum number of iterations taken for the solvers to converge.
#     'tol': (1e-6, 1e-2, 'log-uniform')  # Tolerance for stopping criteria.
# }

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
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False, test_size=0.3,
                 already_splitted_data=None, sampling_strategy='conditionalOversampling', *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                         test_size=test_size, already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)
        # sc=StandardScaler()
        # self.X_train=sc.fit_transform(self.X_train)
        # self.X_test=sc.transform(self.X_test)
        self.estimator = LogisticRegression(*args, **kwargs)



    @property
    def default_params(self):
        return DEFAULT_PARAMS
