from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

# Default parameters for Linear Discriminant Analysis
DEFAULT_PARAMS_LDA = {
    'solver': (['svd', 'lsqr', 'eigen']),  # Solver types
}

class LinearDiscriminantAnalysisClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LinearDiscriminantAnalysis(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS_LDA
