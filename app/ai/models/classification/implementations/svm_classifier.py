from sklearn.svm import SVC
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
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
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                 create_transformations=False, apply_transformations=False, test_size=0.3, already_splitted_data=None, sampling_strategy='conditionalOversampling', *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size, 
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                         create_transformations=create_transformations, apply_transformations=apply_transformations,
                         already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)

        self.estimator = SVC(*args, probability=True, *args, **kwargs) 

    @property
    def default_params(self):
        return DEFAULT_PARAMS
