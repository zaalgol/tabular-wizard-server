from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from xgboost import plot_importance
from src.data_preprocessing import DataPreprocessing
from src.models.regression.implementations.base_regressor_model import BaseRegressorModel


DEFAULT_PARAMS = {
            'max_depth': (3, 10, 1),
            'learning_rate': (0.01, 0.3, "log-uniform"),
            'subsample': (0.5, 1.0, "uniform"),
            "gamma": (1e-9, 0.5, "log-uniform"),
            'colsample_bytree': (0.5, 1.0, "uniform"),
            'colsample_bylevel': (0.5, 1.0, "uniform"),
            'n_estimators': (100, 1000),
            'alpha': (0, 1),
            'lambda': (0, 1),
            'min_child_weight': (1, 10)
        }

class XgboostRegressor(BaseRegressorModel):
    def __init__(self, train_df, target_column, split_column=None,
                 create_encoding_rules=False, apply_encoding_rules=False,
                 test_size=0.3, already_splitted_data=None, scoring='r2', *args, **kwargs):
        
        super().__init__(train_df, target_column, split_column=split_column, test_size=test_size,
                         create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules,
                         already_splitted_data=already_splitted_data, scoring=scoring, *args, **kwargs)
        
        self.X_train = DataPreprocessing().set_not_numeric_as_categorial(self.X_train)
        self.estimator = XGBRegressor(enable_categorical=True, *args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def plot(self, result):
        plot_importance(result.best_estimator_)
        plt.show()

# class XgboostRegressor(BaseModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def tune_hyper_parameters(self, params_constrained=None, hyperparams=None,
#                                              tree_method = "hist",device = None,  early_stopping_rounds=10, eval_metric='rmse',
#                                              scoring='neg_mean_squared_error', n_iter = 25, verbose = 0):
#         if hyperparams is None:
#             self.hyperparams = DEFAULT_PARAMS
#         else:
#             self.hyperparams = hyperparams

#         xgbr = xgb.XGBRegressor(enable_categorical=True, tree_method = tree_method, device = device,
#                                 early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric,
#                                 interaction_constraints=params_constrained
#                                 )
#         kfold = KFold(n_splits=10)
#         self.search = BayesSearchCV(estimator=xgbr,
#                                        search_spaces=self.hyperparams,
#                                        scoring=scoring,
#                                        n_iter=n_iter,
#                                        cv=kfold,
#                                        verbose=verbose)
        

#     def train (self):
#         eval_set = [(self.X_test, self.y_test)]
#         print(self.X_test)
#         result = self.search.fit(self.X_train, self.y_train,
#                                   eval_set=eval_set)  
#         print ("Best parameters:", self.search.best_params_)
#         print ("Lowest RMSE: ", (-self.search.best_score_) ** (1 / 2.0))
#         return result


