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
        df_copy.columns = df_copy.columns.str.replace(' ', '_')
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

        # if model_details.model_type == 'classification':
        #     df_copy = self._process_classification_model(loaded_model, model_details, X_data)

        #     # y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
        #     # df_copy[f'{model_details.target_column}_predict'] = y_predict
        #     # y_predict_proba = self.classificationEvaluate.predict_proba(loaded_model, X_data)
        #     # proba_df = pd.DataFrame(y_predict_proba.round(2), columns=[f'Prob_{cls}' for cls in loaded_model.classes_])
        #     # df_copy = pd.concat([df_copy, proba_df], axis=1)
        #     # df_copy = self.__evaluate_inference(model_details, df_copy, y_predict, y_predict_proba)

        # elif model_details.model_type == 'regression':
        #      df_copy = self._process_regresion_model(loaded_model, model_details, X_data)
        
             
        #     # y_predict = self.regressionEvaluate.predict(loaded_model, X_data)
        #     # df_copy[f'{model_details.target_column}_predict'] = y_predict
        #     # df_copy = self.__evaluate_inference(model_details, df_copy, y_predict, None)

        is_inference_successfully_finished = True

        return (X_data, is_inference_successfully_finished)
    
    # def _process_classification_model(self, loaded_model, model_details, X_data):
    #     y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
    #     df_copy[f'{model_details.target_column}_predict'] = y_predict
    #     y_predict_proba = self.classificationEvaluate.predict_proba(loaded_model, X_data)
    #     proba_df = pd.DataFrame(y_predict_proba.round(2), columns=[f'Prob_{cls}' for cls in loaded_model.classes_])
    #     df_copy = pd.concat([df_copy, proba_df], axis=1)
    #     df_copy = self.__evaluate_inference(model_details, df_copy, y_predict, y_predict_proba)
    #     return df_copy
         
    # def _process_regresion_model(self, loaded_model, model_details, X_data):
    #     y_predict = self.regressionEvaluate.predict(loaded_model, X_data)
    #     df_copy[f'{model_details.target_column}_predict'] = y_predict
    #     df_copy = self.__evaluate_inference(model_details, df_copy, y_predict, None)
    #     return df_copy
         


    # def run_pre_training_data_pipeline(self, model, df):
    #     model.file_line_num = len(df)
    #     # df = self.__dataset_to_df(dataset)
    #     df = self.__data_processing_before_spliting(df, model)
    #     X_train, X_test, y_train, y_test = self.__split_data(df, model)
    #     if model.model_type == 'classification':
    #         self.is_multi_class = DataPreprocessing().get_class_num(y_train) > 2
    #     X_train, X_test, y_train, embedding_rules, encoding_rules, transformations = self.__data_processing_after_spliting(X_train, X_test, y_train, model)

    #     return X_train, X_test, y_train, y_test, embedding_rules, encoding_rules, transformations
        

    # # def __dataset_to_df(self, dataset):
    # #     headers = dataset[0]
    # #     data_rows = dataset[1:]
    # #     df = pd.DataFrame(data_rows, columns=headers)
    # #     return df
    
    # def __data_processing_before_spliting(self, df, model):
    #     self.data_preprocessing.delete_empty_rows(df, model.target_column)
    #     if model.model_type == 'regression':
    #         self.data_preprocessing.delete_rows_with_categorical_target_column(df, model.target_column)
    #     if model.is_time_series:
    #         df, model.time_series_code = self.llm_task.use_llm_to_proccess_timeseries_dataset(df)
    #     if model.training_strategy in ['ensembleModelsFast', 'ensembleModelsTuned']:
    #         df = self.data_preprocessing.fill_missing_numeric_cells(df)
    #     df = self.data_preprocessing.convert_datetime_columns_to_datetime_dtype(df, model)
        
    #     return df 
    
    # def __split_data(self, df, model):
    #         X_train, X_test, y_train, y_test = train_test_split(df,
    #                                                             df[model.target_column], shuffle= not model.time_series_code,
    #                                                             test_size=Config.DATASET_SPLIT_SIZE, random_state=42)
    #         return X_train, X_test, y_train, y_test
    
    # def __data_processing_after_spliting(self, X_train, X_test, y_train, model):
    #     embedding_rules = None
    #     encoding_rules = None 
    #     transformations = None 
    #     X_train = DataPreprocessing().set_not_numeric_as_categorial(X_train)
    #     X_test = DataPreprocessing().set_not_numeric_as_categorial(X_test)
    #     semantic_columns = [k for k, v in model.columns_type.items() if v=='semantic']
    #     if semantic_columns:
    #         embedding_rules = self.nlp_embeddings_preprocessing.create_embedding_rules(X_train, semantic_columns)
    #         X_train = self.nlp_embeddings_preprocessing.apply_embedding_rules(X_train, embedding_rules)
    #         X_test = self.nlp_embeddings_preprocessing.apply_embedding_rules(X_test, embedding_rules)

    #     if model.training_strategy in ['ensembleModelsFast', 'ensembleModelsTuned']:
    #         encoding_rules = self.data_preprocessing.create_encoding_rules(X_train)
    #         X_train = self.data_preprocessing.apply_encoding_rules(X_train, encoding_rules)
    #         X_test = self.data_preprocessing.apply_encoding_rules(X_test, encoding_rules)

    #         numeric_columns = self.data_preprocessing.get_numeric_columns(X_train)
    #         transformations = self.data_preprocessing.create_transformed_numeric_column_details(X_train, numeric_columns)
    #         X_train = self.data_preprocessing.transformed_numeric_column_details(X_train, transformations)
    #         X_test = self.data_preprocessing.transformed_numeric_column_details(X_test, transformations)

    #         X_train = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(X_train)
    #         X_test = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(X_test)


    #     X_train = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(X_train)
    #     X_test = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(X_test)
    #     if model.model_type == 'classification':
    #         if model.sampling_strategy == 'conditionalOversampling':
    #             self.apply_conditional_oversampling()
    #         elif model.sampling_strategy == 'oversampling':
    #             self.apply_oversampling(X_train, y_train)

    #     return X_train, X_test, y_train, embedding_rules, encoding_rules, transformations
    
    # def apply_conditional_oversampling(self,X_train, y_train):
    #     class_counts = y_train.value_counts()

    #     smallest_class = class_counts.min()
    #     largest_class = class_counts.max()
    #     ratio = smallest_class / largest_class

    #     imbalance_threshold = Config.IMBALACE_THRESHOLD

    #     # If the ratio is below the threshold, apply oversampling
    #     if ratio < imbalance_threshold:
    #         return self.apply_oversampling(X_train, y_train)
    #     else:
    #         print("The dataset is considered balanced. Skipping oversampling.")
    #         return X_train, y_train
                
    # def apply_oversampling(self, X_train, y_train):
    #     class_counts = y_train.value_counts()
    #     max_size = class_counts.max()

    #     X_train_resampled = []
    #     y_train_resampled = []

    #     for class_index, count in class_counts.items():
    #         df_class_indices = y_train[y_train == class_index].index
    #         df_class = X_train.loc[df_class_indices]
    #         if count < max_size:
    #             df_class_over = resample(df_class, 
    #                                      replace=True,  # sample with replacement
    #                                      n_samples=max_size,  # match number in majority class
    #                                      random_state=42)  # reproducible results
    #             y_class_over = resample(y_train.loc[df_class_indices], 
    #                                     replace=True, 
    #                                     n_samples=max_size, 
    #                                     random_state=42)
    #             X_train_resampled.append(df_class_over)
    #             y_train_resampled.append(y_class_over)
    #         else:
    #             X_train_resampled.append(df_class)
    #             y_train_resampled.append(y_train.loc[df_class_indices])
        
    #     X_train = pd.concat(X_train_resampled)
    #     y_train = pd.concat(y_train_resampled)
