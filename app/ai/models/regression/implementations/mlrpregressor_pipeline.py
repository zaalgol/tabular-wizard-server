import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# https://chat.openai.com/c/6de8cee9-dbd1-4f79-a0b1-57d7303e78d5
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, func=None):
        self.func = func
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.func:
            return self.func(X.copy())
        return X

class ColumnRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_remove=None):
        self.columns_to_remove = columns_to_remove
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.columns_to_remove:
            result =  X.drop(columns=self.columns_to_remove,  errors='ignore')
            return result
        return X
    
class Mlrpegressor:
    def __init__(self, dataframe, target_column, scale=False, order_columns_and_values=None,
                    columns_to_remove=[], feature_engineering_func=None):
        self.dataframe = dataframe
        self.target_column = target_column
        self.scale = scale
        self.order_columns_and_values = order_columns_and_values
        self.columns_to_remove = columns_to_remove
        self.feature_engineering_func = feature_engineering_func
    
    def train_model(self):
        X = self.dataframe.drop(columns=[self.target_column])
        y = self.dataframe[self.target_column]

        num_features = X.select_dtypes(exclude=['object']).columns.tolist()
        num_features = [col for col in num_features if col not in self.columns_to_remove]
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
        cat_features = [col for col in cat_features if col not in self.columns_to_remove]

        if self.order_columns_and_values:
            cat_features_ordinal = list(self.order_columns_and_values.keys())
            cat_features_onehot = list(set(cat_features) - set(cat_features_ordinal))
            ordinal_categories = list(self.order_columns_and_values.values())
        else:
            cat_features_ordinal = []
            cat_features_onehot = cat_features
            ordinal_categories = []

        num_transformer_steps = [
            ('num_imputer', SimpleImputer(strategy='mean'))
        ]

        cat_transformer_onehot_steps = [
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
        
        if self.scale:
            num_transformer_steps.append(('scaler', StandardScaler()))
            # cat_transformer_onehot_steps.append(('scaler', StandardScaler()))
        num_transformer = Pipeline(steps=num_transformer_steps)

        cat_transformer_onehot = Pipeline(steps = cat_transformer_onehot_steps)

        cat_transformer_ordinal = Pipeline(steps=[
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=ordinal_categories))
        ])

        transformers=[
            ('num', num_transformer, num_features),
            ('cat_onehot', cat_transformer_onehot, cat_features_onehot)
        ]
        if self.order_columns_and_values:
            transformers.append(('cat_ordinal', cat_transformer_ordinal, cat_features_ordinal))
        preprocessor = ColumnTransformer(transformers)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        n_cols = self.X_train.shape[1]
        hidden_layer_sizes = (int(np.sqrt(n_cols + 1)),)
        
        pipeline_steps = [
            ('feature_engineer', FeatureEngineer(func=self.feature_engineering_func)),
            ('column_remover', ColumnRemover(columns_to_remove=self.columns_to_remove)),
            ('preprocessor', preprocessor),
            ('nn_regressor', MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes, 
                max_iter=100000,
                random_state=42,
                solver='adam',
                alpha=0.5 
            ))
        ]
        model = Pipeline(steps=pipeline_steps)

        return model.fit(self.X_train, self.y_train)
        