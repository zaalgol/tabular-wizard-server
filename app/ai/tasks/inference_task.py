import re
import numpy as np
import pandas as pd
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
from app.ai.data_preprocessing import DataPreprocessing
from app.ai.nlp_embeddings_preprocessing import NlpEmbeddingsPreprocessing
from app.ai.pipelines.inference_pipeline import InferencePipeline
from app.ai.tasks.llm_task import LlmTask

class InferenceTask:
    def __init__(self) -> None:
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.nlp_embeddings_preprocessing = NlpEmbeddingsPreprocessing()
        self.llm_task = LlmTask()
        self.inferencePipeline = InferencePipeline()

    def run_task(self, model_details, loaded_model, inference_df):
        try:
            is_inference_successfully_finished = False
            
            X_data, is_inference_successfully_finished = self.inferencePipeline.pre_process(loaded_model, model_details, inference_df)

            # df_copy = inference_df.copy()
            # if model_details.is_time_series:
            #     df_copy = self.llm_task.processed_dataset(inference_df.copy(), model_details.time_series_code)
            # X_data = self.data_preprocessing.exclude_columns(df_copy, columns_to_exclude=[model_details.target_column])
            # X_data = self.__data_preprocessing(X_data, model_details)
            
            if model_details.model_type == 'classification':
                y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
                inference_df[f'{model_details.target_column}_predict'] = y_predict
                y_predict_proba = self.classificationEvaluate.predict_proba(loaded_model, X_data)
                proba_df = pd.DataFrame(y_predict_proba.round(2), columns=[f'Prob_{cls}' for cls in loaded_model.classes_])
                inference_df = pd.concat([inference_df, proba_df], axis=1)
                inference_df = self.__evaluate_inference(model_details, inference_df, y_predict, y_predict_proba)

            elif model_details.model_type == 'regression':
                y_predict = self.regressionEvaluate.predict(loaded_model, X_data)
                inference_df[f'{model_details.target_column}_predict'] = y_predict
                inference_df = self.__evaluate_inference(model_details, inference_df, y_predict, None)
                
                
            is_inference_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            return (model_details, inference_df, is_inference_successfully_finished)

    # def __data_preprocessing(self, df, model):
    #     df_copy = df.copy()
    #     df_copy = self.data_preprocessing.sanitize_cells(df_copy)
    #     df_copy = self.data_preprocessing.fill_missing_numeric_cells(df_copy)
    #     df_copy = self.data_preprocessing.set_not_numeric_as_categorial(df_copy)
    #     df_copy = self.data_preprocessing.convert_datetime_columns_to_datetime_dtype(df_copy, model)
    #     if model.encoding_rules:
    #         df_copy = self.data_preprocessing.apply_encoding_rules(df_copy, model.encoding_rules)
    #     if model.embedding_rules:
    #         df_copy = self.nlp_embeddings_preprocessing.apply_embedding_rules(df_copy, model.embedding_rules)
    #     if model.transformations:
    #          df_copy = self.data_preprocessing.transformed_numeric_column_details(df_copy, model.transformations)
    #     df_copy = self.data_preprocessing.convert_datatimes_columns_to_normalized_floats(df_copy)
    #     return df_copy
    
    def extract_original_metrics(self, metrics_string, key):
        # Define regex pattern to capture the value for the given key
        pattern = fr'{key}: ([\d\.\-e]+)'
        
        # Find all occurrences of the key in the string
        matches = re.findall(pattern, metrics_string)
        
        if len(matches) < 2:
            raise ValueError(f'Could not find both train and test values for key: {key}')
        
        # Return the first match as the train value and the second as the test value
        return matches[0], matches[1]
    
    def format_evaluation(self, value):
        try:
            return f"{float(value):.3f}"
        except ValueError:
            return value
    
    
    def __evaluate_inference(self, model_details, inference_df, y_predict, y_predict_proba):
        if model_details.target_column in inference_df.columns:
            filtered_original, filtered_predicted, filtered_y_predict_proba = \
                self.data_preprocessing.filter_invalid_entries(inference_df[model_details.target_column], y_predict, y_predict_proba)
            if model_details.model_type == 'classification':
                # TODO: use filtered_original, filtered_predicted and also filtered_y_predict
                inference_evaluations = self.classificationEvaluate.calculate_metrics(filtered_original, filtered_predicted, filtered_y_predict_proba)
            elif model_details.model_type == 'regression':
                inference_evaluations = \
                    self.regressionEvaluate.calculate_metrics(filtered_original, filtered_predicted)

            # Prepare the evaluation data
            # Add 'Evaluations' column to the original DataFrame
            if 'Evaluations' not in inference_df.columns:
                # inference_df.insert(0, 'Evaluations', np.nan)
                inference_df["Evaluations"] = np.nan

                # Prepare the evaluation data
                eval_types = ['Inference:', 'Train:', 'Test:']
                eval_data = {'Evaluations': eval_types}

                # Create empty columns in inference_df for each evaluation metric
                for key in inference_evaluations.keys():
                    train_eval, test_eval = self.extract_original_metrics(model_details.formated_evaluations, key)
                    if key not in inference_df.columns:
                        inference_df[key] = np.nan

                    eval_data[key] = [
                        self.format_evaluation(inference_evaluations[key]),
                        self.format_evaluation(train_eval),
                        self.format_evaluation(test_eval)
                    ]

                # Create eval_df with only the evaluation metrics
                eval_df = pd.DataFrame(eval_data)

                # Iterate over eval_df rows and cells to copy the values into inference_df
                for row_index, row in eval_df.iterrows():
                    for column_name, cell_value in row.items():
                        inference_df.at[row_index, column_name] = cell_value

        return inference_df


