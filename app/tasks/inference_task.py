import pandas as pd
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
from app.ai.data_preprocessing import DataPreprocessing


class InferenceTask:
    def __init__(self) -> None:
        self.data_preprocessing = DataPreprocessing()
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()

    def run_task(self, model_details, loaded_model, original_df, inference_task_callback, app_context):
        try:
            is_inference_successfully_finished = False
            X_data = self.data_preprocessing.exclude_columns(original_df, columns_to_exclude=[model_details.target_column]).copy()
            X_data = self._data_preprocessing(X_data, model_details.encoding_rules)
            
            if model_details.model_type == 'classification':
                y_predict = self.classificationEvaluate.predict(loaded_model, X_data)
                original_df[f'{model_details.target_column}_predict'] = y_predict
                y_predict_proba = self.classificationEvaluate.predict_proba(loaded_model, X_data)
                proba_df = pd.DataFrame(y_predict_proba.round(2), columns=[f'Prob_{cls}' for cls in loaded_model.classes_])
                original_df = pd.concat([original_df, proba_df], axis=1)

            elif model_details.model_type == 'regression':
                y_predict = self.regressionEvaluate.predict(loaded_model, X_data)
                original_df[f'{model_details.target_column}_predict'] = y_predict
                
            is_inference_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            inference_task_callback(model_details, original_df, is_inference_successfully_finished, app_context)

    def _data_preprocessing(self, df, encoding_rules):
        df_copy = df.copy()
        df_copy = self.data_preprocessing.sanitize_cells(df_copy)
        df_copy = self.data_preprocessing.fill_missing_numeric_cells(df_copy)
        df_copy = self.data_preprocessing.set_not_numeric_as_categorial(df)
        if encoding_rules:
            df_copy = self.data_preprocessing.apply_encoding_rules(df_copy, encoding_rules)
        return df_copy
    
    def _evaluate_inference(self, model_details, original_df):
        # if not original_df[model_details.target_column].empty:
        # original_df['model_accuracy'] = np.nan
        #     original_df.at[0, 'model_accuracy'] = accuracy_score(original_df[model_details.target_column].values, y_predict)
        pass



