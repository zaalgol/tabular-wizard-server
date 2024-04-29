from sklearn.svm import SVR
from src.data_preprocessing import DataPreprocessing
from src.models.regression.implementations.base_regressor_model import BaseRegressorModel

DEFAULT_PARAMS_SVR = {
    'C': (0.1, 1000, "log-uniform"),
    'epsilon': (0.01, 1.0, "uniform"),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': (1, 5)  # Only relevant for poly kernel
}

class SVRRegressorModel(BaseRegressorModel):
    def __init__(self, train_df, target_column, split_column=None,
                 create_encoding_rules=False, apply_encoding_rules=False,
                 create_transformations=False, apply_transformations=False, 
                 test_size=0.3, already_splitted_data=None, scoring='r2', *args, **kwargs):
        
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         create_transformations=create_transformations, apply_transformations=apply_transformations,
                         already_splitted_data=already_splitted_data, scoring=scoring, *args, **kwargs)
        
        self.estimator = SVR(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS_SVR
