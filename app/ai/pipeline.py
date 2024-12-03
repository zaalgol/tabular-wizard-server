import copy
import pandas as pd
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import optuna
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.nlp_embeddings_preprocessing import NlpEmbeddingsPreprocessing
from app.ai.data_preprocessing import DataPreprocessing
from app.config.config import Config
from app.tasks.llm_task import LlmTask


class Pipeline:
    def __init__(self) -> None:
        self.llm_task = LlmTask()
        self.data_preprocessing = DataPreprocessing()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
            
    def run_pre_training_data_pipeline(self, model, dataset, split_column=None):
        model.file_line_num = len(dataset)
        df = self.__dataset_to_df(dataset)
        self.__data_processing_before_spliting(df, model)
        X_train, X_test, y_train, y_test = self.__split_data(self, df, split_column, model.target_column)
        embedding_rules, encoding_rules, transformations = self.__data_processing_after_spliting(X_train, X_test, model)

        return X_train, X_test, y_train, y_test, embedding_rules, encoding_rules, transformations
        

    def __dataset_to_df(self, dataset):
        headers = dataset[0]
        data_rows = dataset[1:]
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    
    def __data_processing_before_spliting(self, df, model, fill_missing_numeric_cells=True):
        if model.is_time_series:
            df, model.time_series_code = self.llm_task.use_llm_toproccess_timeseries_dataset(df)
        if fill_missing_numeric_cells:
            df = self.data_preprocessing.fill_missing_numeric_cells(df)
        df = self.data_preprocessing.convert_datetime_columns_to_datetime_dtype(df, model)
        self.data_preprocessing.delete_empty_rows(df, model.target_column)
        if model.model_type == 'regression':
            self.data_preprocessing.delete_non_numeric_rows(df, model.target_column)
        return df 
    
    def __split_data(self, df, target_column, shuffle=True):
            X_train, X_test, y_train, y_test = train_test_split(df,
                                                                df[target_column], shuffle=shuffle,
                                                                test_size=Config.DATASET_SPLIT_SIZE, random_state=42)
            return X_train, X_test, y_train, y_test
    
    def __data_processing_after_spliting(self, X_train, X_test, model):
        semantic_columns = [k for k, v in model.columns_type.items() if v=='semantic']
        if semantic_columns:
            embedding_rules = self.nlp_embeddings_preprocessing.create_embedding_rules(X_train, semantic_columns)
            X_train = self.nlp_embeddings_preprocessing.apply_embedding_rules(X_train, embedding_rules)
            X_test = self.nlp_embeddings_preprocessing.apply_embedding_rules(X_test, embedding_rules)

        if model.training_strategy in ['ensembleModelsFast', 'ensembleModelsTuned']:
            encoding_rules = self.data_preprocessing.create_encoding_rules(X_train)
            X_train = self.data_preprocessing.apply_encoding_rules(X_train, encoding_rules)
            X_test = self.data_preprocessing.apply_encoding_rules(X_test, encoding_rules)

            numeric_columns = self.data_preprocessing.get_numeric_columns(X_train)
            transformations = self.data_preprocessing.create_transformed_numeric_column_details(X_train, numeric_columns)
            X_train = self.data_preprocessing.transformed_numeric_column_details(X_train, transformations)
            X_test = self.data_preprocessing.transformed_numeric_column_details(X_test, transformations)

            X_train = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(X_train)
            X_test = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(X_test)

        return embedding_rules, encoding_rules, transformations

        

        


    
