from abc import abstractmethod
import pprint
import time
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import optuna
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.nlp_embeddings_preprocessing import NlpEmbeddingsPreprocessing


class BaseModel:
    def __init__(self, train_df, target_column):
        self.train_df = train_df
        self.study = None
        self.target_column = target_column
        self.data_preprocessing = DataPreprocessing()
        
        # self.__split_data(split_column, test_size)

    # def __split_data(self, split_column, test_size):
    #     # if already_splitted_data:
    #     #     self.X_train, self.X_test, self.y_train, self.y_test = \
    #     #         already_splitted_data['X_train'], already_splitted_data['X_test'], already_splitted_data['y_train'], already_splitted_data['y_test']
    #     #     return

    #     if split_column is None:
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train_df,
    #                                                                                 self.train_df[self.target_column],
    #                                                                                 test_size=test_size, random_state=42)
    #     else:
    #         splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7)
    #         split = splitter.split(self.train_df, groups=self.train_df[split_column])
    #         train_inds, test_inds = next(split)

    #         train = self.train_df.iloc[train_inds]
    #         self.y_train = train[[self.target_column]].astype(float)
    #         test = self.train_df.iloc[test_inds]
    #         self.y_test = test[[self.target_column]].astype(float)

    #     self.X_train = self.X_train.drop([self.target_column], axis=1)
    #     self.X_test = self.X_test.drop([self.target_column], axis=1)

        # self._preprocess_data(create_encoding_rules, apply_encoding_rules, create_transformations, apply_transformations)



    # def _preprocess_data(self, create_encoding_rules, apply_encoding_rules, create_transformations, apply_transformations,
    #                            create_embedding_rules=True, apply_embedding_rules=True):
        
    #     if create_embedding_rules and self.semantic_columns:
    #         self.embedding_rules = self.nlp_embeddings_preprocessing.create_embedding_rules(self.X_train, self.semantic_columns)

    #     if apply_embedding_rules and self.semantic_columns:
    #         self.X_train = self.nlp_embeddings_preprocessing.apply_embedding_rules(self.X_train, self.embedding_rules)
    #         self.X_test = self.nlp_embeddings_preprocessing.apply_embedding_rules(self.X_test, self.embedding_rules)
            
    #     if create_encoding_rules:
    #         self.encoding_rules = self.data_preprocessing.create_encoding_rules(self.X_train)

    #     if apply_encoding_rules:
    #         self.X_train = self.data_preprocessing.apply_encoding_rules(self.X_train, self.encoding_rules)
    #         self.X_test = self.data_preprocessing.apply_encoding_rules(self.X_test, self.encoding_rules)

    #     if create_transformations:
    #         numeric_columns = self.data_preprocessing.get_numeric_columns(self.X_train)
    #         self.transformations = self.data_preprocessing.create_transformed_numeric_column_details(self.X_train, numeric_columns)

    #     if apply_transformations:
    #         self.X_train = self.data_preprocessing.transformed_numeric_column_details(self.X_train, self.transformations)
    #         self.X_test = self.data_preprocessing.transformed_numeric_column_details(self.X_test, self.transformations)
        

        
    #     self.X_train = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(self.X_train)
    #     self.X_test = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(self.X_test)

    def remove_unnecessary_parameters_for_implementations(self, kwargs):
        for parameter in self.unnecessary_parameters:
            kwargs.pop(parameter, None)
            
    @property
    def default_values(self):
        return {}

    @property
    def callbacks(self):
        # Optuna handles time limits via study.optimize() parameters
        return []

    @property
    @abstractmethod
    def unnecessary_parameters(self):
        return {}

    @property
    @abstractmethod
    def default_params(self):
        return {}
