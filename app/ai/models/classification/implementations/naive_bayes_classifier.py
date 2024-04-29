from sklearn.naive_bayes import GaussianNB
from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
from skopt.space import Real

# Default parameters for Gaussian Naive Bayes
DEFAULT_PARAMS = {
    'var_smoothing': Real(1e-9, 1e-6, prior='log-uniform'),  # Portion of the largest variance of all features to be added to variances for stability
}

class NaiveBayesClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                 create_transformations=False, apply_transformations=False, test_size=0.3, already_splitted_data=None, 
                 sampling_strategy='conditionalOversampling', *args, **kwargs):
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size, 
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         create_transformations=create_transformations, apply_transformations=apply_transformations,
                         already_splitted_data=already_splitted_data, sampling_strategy=sampling_strategy, *args, **kwargs)

        self.estimator = GaussianNB(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
