from sklearn.naive_bayes import GaussianNB
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

# Default parameters for Gaussian Naive Bayes
DEFAULT_PARAMS = {
    'var_smoothing': (1e-9, 1e-6, 'log-uniform'),  # Portion of the largest variance of all features to be added to variances for stability
}

class GaussianNaiveBayesClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = GaussianNB(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
