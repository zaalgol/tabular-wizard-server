import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel

class LinearDiscriminantAnalysisClassifier(BaseClassfierModel):
    def __init__(self, train_df, target_column, *args, **kwargs):
        super().__init__(train_df, target_column, *args, **kwargs)
        self.remove_unnecessary_parameters_for_implementations(kwargs)
        self.estimator = LinearDiscriminantAnalysis(*args, **kwargs)

    @property
    def default_params(self):
        n_features = self.X_train.shape[1]
        n_classes = len(np.unique(self.y_train))
        max_components = min(n_features, n_classes - 1)

        # Calculate balanced priors
        class_counts = np.bincount(self.y_train)
        balanced_priors = class_counts / len(self.y_train)

        return {
            'solver': ['svd', 'lsqr', 'eigen'],  # Solver types
            # 'shrinkage': [None, 'auto'] + list(np.arange(0.0, 1.01, 0.1)),  # Shrinkage parameter
            'tol': (1e-5, 1e-3, 'log-uniform'),  # Threshold used for rank estimation
            'n_components': (1, max_components, 'int'),  # Number of components for dimensionality reduction
            'store_covariance': [True, False],  # Whether to compute and store covariance matrices
            'priors': [None, balanced_priors.tolist()],   # Prior probabilities of the classes
        }
