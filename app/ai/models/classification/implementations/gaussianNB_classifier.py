from sklearn.naive_bayes import GaussianNB
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

# Default parameters for Gaussian Naive Bayes
GAUSSIAN_DEFAULT_PARAMS = {
    'var_smoothing': (1e-10, 1e-8, 'log-uniform'),  # Portion of the largest variance of all features to be added to variances for stability
    'priors': [None, 'balanced']  # Prior probabilities of the classes
}

class GaussianNaiveBayesClassifier(BaseClassfierModel):
    def __init__(self, target_column, scoring, *args, **kwargs):
        super().__init__(target_column, scoring, *args, **kwargs)
        # self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = GaussianNB(*args, **kwargs)

    @property
    def default_params(self):
        return GAUSSIAN_DEFAULT_PARAMS
