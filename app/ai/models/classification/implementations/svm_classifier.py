from sklearn.svm import SVC
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Categorical, Real, Integer

# Default parameters for SVM
DEFAULT_PARAMS = {
    'C': Real(0.1, 10.0),  # Regularization parameter
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),  # Specifies the kernel type to be used in the algorithm
    'degree': Integer(2, 5),  # Degree of the polynomial kernel function (ignored by all other kernels)
    'gamma': Categorical(['scale', 'auto']),  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
    'coef0': Real(0.0, 10.0),  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
}

class SvmClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = SVC(*args, probability=True, *args, **kwargs) 

    @property
    def default_params(self):
        return DEFAULT_PARAMS
