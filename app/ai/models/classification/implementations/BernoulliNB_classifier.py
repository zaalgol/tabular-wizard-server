from sklearn.naive_bayes import BernoulliNB
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

# Default parameters for Bernoulli Naive Bayes
DEFAULT_PARAMS = {
    'alpha': (1e-3, 1e0, "log-uniform"),  # Additive (Laplace/Lidstone) smoothing parameter
    'binarize': (0, 1, "uniform"),       # Threshold for binarizing (mapping to {0, 1}) input features
}

class BernoulliNaiveBayesClassifier(BaseClassfierModel):
    def __init__(self, target_column, scoring, *args, **kwargs):
        super().__init__(target_column, scoring, *args, **kwargs)
        # self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = BernoulliNB(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
