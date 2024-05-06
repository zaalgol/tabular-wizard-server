from sklearn.svm import NuSVC
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Real, Categorical

# Default parameters for Nu-Support Vector Classification
DEFAULT_PARAMS = {
    'nu': Real(0.01, 1.0, prior='uniform'),  # An upper bound on the fraction of training errors
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),  # Kernel types
    'degree': Real(2, 5, prior='uniform'),  # Degree for polynomial kernel function
    'gamma': Real(1e-6, 1e0, prior='log-uniform'),  # Kernel coefficient
    'coef0': Real(0, 10, prior='uniform'),  # Independent term in kernel function
}

class NuSVCClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = NuSVC(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
