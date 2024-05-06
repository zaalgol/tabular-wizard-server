from sklearn.naive_bayes import BernoulliNB
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Real

# Default parameters for Bernoulli Naive Bayes
DEFAULT_PARAMS = {
    'alpha': Real(1e-3, 1e0, prior='log-uniform'),  # Additive (Laplace/Lidstone) smoothing parameter
    'binarize': Real(0, 1, prior='uniform'),       # Threshold for binarizing (mapping to {0, 1}) input features
}

class BernoulliNaiveBayesClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = BernoulliNB(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
