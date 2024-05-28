from sklearn.svm import NuSVC
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

# Default parameters for Nu-Support Vector Classification
DEFAULT_PARAMS = {
    'nu': (0.01, 1.0, 'uniform'),  # An upper bound on the fraction of training errors
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types
    'degree': (2, 5, 'uniform'),  # Degree for polynomial kernel function
    'gamma': (1e-6, 1e0, 'log-uniform'),  # Kernel coefficient
    'coef0': (0, 10, 'uniform'),  # Independent term in kernel function
}

class NuSVCClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = NuSVC(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
