import numpy as np
from sklearn.linear_model import LogisticRegression
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

DEFAULT_PARAMS = {
    'C': (0.0001, 10, 'log-uniform'),
    'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty' : ['l2'],
    'max_iter' : [100, 1000,2500, 5000]
    }
    
    # TODO: integrate this params to work with our implemetation
    # {
    #     'solver': ['newton-cg', 'lbfgs', 'sag'],
    #     'penalty': ['l2', None],
    #     'class_weight': ['balanced', None],
    #     'C': (0.0001, 10, 'log-uniform'),
    #     'fit_intercept': [True, False],
    #     'max_iter': (100, 1000, 'int'),
    #     'tol': (1e-6, 1e-2, 'log-uniform')
    # },
    # {
    #     'solver': ['liblinear'],
    #     'penalty': ['l1', 'l2'],
    #     'class_weight': ['balanced', None],
    #     'C': (0.0001, 10, 'log-uniform'),
    #     'fit_intercept': [True, False],
    #     'max_iter': (100, 1000, 'int'),
    #     'tol': (1e-6, 1e-2, 'log-uniform')
    # },
    # {
    #     'solver': ['saga'],
    #     'penalty': ['l1', 'l2', None],
    #     'class_weight': ['balanced', None],
    #     'C': (0.0001, 10, 'log-uniform'),
    #     'fit_intercept': [True, False],
    #     'max_iter': (100, 1000, 'int'),
    #     'tol': (1e-6, 1e-2, 'log-uniform')
    # },
    # {
    #     'solver': ['saga'],
    #     'penalty': ['elasticnet'],
    #     'class_weight': ['balanced', None],
    #     'C': (0.0001, 10, 'log-uniform'),
    #     'fit_intercept': [True, False],
    #     'max_iter': (100, 1000, 'int'),
    #     'tol': (1e-6, 1e-2, 'log-uniform'),
    #     'l1_ratio': (0, 1, 'uniform')
    # }




class LRegression(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LogisticRegression(*args, **kwargs)



    @property
    def default_params(self):
        return DEFAULT_PARAMS
