from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

# Default parameters for Quadratic Discriminant Analysis
DEFAULT_PARAMS_QDA = {
    'reg_param':(0.0, 1.0, 'uniform'),  # Regularization parameter
}

class QuadraticDiscriminantAnalysisClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = QuadraticDiscriminantAnalysis(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS_QDA
