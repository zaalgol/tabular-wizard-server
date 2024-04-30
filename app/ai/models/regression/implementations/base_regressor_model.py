from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from app.ai.models.base_model import BaseModel


class BaseRegressorModel(BaseModel):
        def __init__(self, train_df, target_column,  scoring='r2',
                       *args, **kwargs):
            super().__init__(train_df, target_column, scoring, *args, **kwargs)

        def tune_hyper_parameters(self, params=None, kfold=5, n_iter=50, *args,**kwargs):
            if params is None:
                params = self.default_params
            Kfold = KFold(n_splits=kfold) 
            
            self.search = BayesSearchCV(estimator=self.estimator,
                                        search_spaces=params,
                                        scoring=self.scoring,
                                        n_iter=n_iter,
                                        n_jobs=1, 
                                        n_points=3,
                                        cv=Kfold,
                                        verbose=0,
                                        random_state=0)
            
        def train(self):
            if self.search:
                result = self.search.fit(self.X_train, self.y_train)
                print("Best parameters:", self.search.best_params_)
                print("Lowest RMSE: ", (-self.search.best_score_) ** (1 / 2.0))
            else:
                result = self.estimator.fit(self.X_train, self.y_train)
            return result
        
        @property
        def unnecessary_parameters(self):
            return ['scoring', 'split_column', 'create_encoding_rules', 'apply_encoding_rules', 'create_transformations', 'apply_transformations', 'test_size',
                    'already_splitted_data']
        
        

                