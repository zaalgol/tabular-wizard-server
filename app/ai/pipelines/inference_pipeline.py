import copy
import pandas as pd
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import optuna
from sklearn.utils import resample
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.nlp_embeddings_preprocessing import NlpEmbeddingsPreprocessing
from app.ai.data_preprocessing import DataPreprocessing
from app.config.config import Config
from app.ai.tasks.llm_task import LlmTask


class InferencePipeline:
    _instance = None

    def __init__(self) -> None:
        self.llm_task = LlmTask()
        self.data_preprocessing = DataPreprocessing()
        self.nlp_embeddings_preprocessing = NlpEmbeddingsPreprocessing()

    def pre_process(self, loaded_model, model_details, inference_df):
        is_inference_successfully_finished = False
        df_copy = inference_df.copy()
        if model_details.is_time_series:
            df_copy = self.llm_task.processed_dataset(df_copy, model_details.time_series_code)
        X_data = self.data_preprocessing.exclude_columns(df_copy, columns_to_exclude=[model_details.target_column])
        X_data = self.data_preprocessing.fill_missing_numeric_cells(X_data)
        X_data = self.data_preprocessing.set_not_numeric_as_categorial(X_data)
        X_data = self.data_preprocessing.convert_datetime_columns_to_datetime_dtype(X_data, model_details)
        if model_details.encoding_rules:
            X_data = self.data_preprocessing.apply_encoding_rules(X_data, model_details.encoding_rules)
        if model_details.embedding_rules:
            X_data = self.nlp_embeddings_preprocessing.apply_embedding_rules(X_data, model_details.embedding_rules)
        if model_details.transformations:
             X_data = self.data_preprocessing.transformed_numeric_column_details(X_data, model_details.transformations)
        X_data = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(X_data)

        X_data = X_data[loaded_model.feature_names_in_]

        is_inference_successfully_finished = True

        return (X_data, is_inference_successfully_finished)
    