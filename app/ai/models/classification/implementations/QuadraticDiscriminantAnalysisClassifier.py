from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

# Default parameters for Quadratic Discriminant Analysis
DEFAULT_PARAMS_QDA = {
    'reg_param': (0.0, 1.0, 'uniform'),  # Regularization parameter
    'tol': (1e-4, 1e-2, 'log-uniform'),  # Threshold used for rank estimation
    'store_covariance': [True, False],  # Whether to compute and store covariance matrices
    # 'priors': [None, 'balanced'],  # Prior probabilities of the classes # getting error"The 'priors' parameter of QuadraticDiscriminantAnalysis must be an array-like or None. Got 'balanced' instead.""
}

class QuadraticDiscriminantAnalysisClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        # self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = QuadraticDiscriminantAnalysis(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS_QDA
