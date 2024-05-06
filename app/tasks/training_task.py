from app.ai.models.classification.implementations.lightgbm_classifier import LightgbmClassifier
from app.ai.data_preprocessing import DataPreprocessing 
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
from app.ai.models.regression.implementations.lightgbm_regerssor import LightGBMRegressor
from app.ai.models.regression.implementations.ensemble import Ensemble as RegressionEnsemble
from app.ai.models.classification.implementations.ensemble import Ensemble as ClassificationEnsemble


class TrainingTask:
    def __init__(self) -> None:
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.data_preprocessing = DataPreprocessing()

    def run_task(self, model, headers, df, task_callback, app_context):
        is_training_successfully_finished = False
        trained_model = None
        evaluations = None
        encoding_rules = None
        transformations = None
        try:
            if model.training_strategy == 'ensembleModelsFast' or model.training_strategy == 'ensembleModelsTuned':
                trained_model, evaluations, encoding_rules, transformations = self.__train_multi_models(model, df)
            else:
                trained_model, evaluations, encoding_rules, transformations = self.__train_single_model(model, df)
            is_training_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            model.evaluations = evaluations
            task_callback(model, trained_model, encoding_rules, transformations,  headers, is_training_successfully_finished, app_context)

    def __train_single_model(self, model, df):
        df = self.__data_preprocessing(df, fill_missing_numeric_cells=True)
        if model.model_type == 'classification':
            training_model = LightgbmClassifier(train_df = df, target_column = model.target_column, scoring=model.metric, 
                                                sampling_strategy=model.sampling_strategy)
            evaluate = self.classificationEvaluate

        elif model.model_type == 'regression':
            training_model = LightGBMRegressor(train_df = df, target_column = model.target_column, scoring=model.metric)
            evaluate = self.regressionEvaluate

        if model.training_strategy == 'singleModelTuned':
            training_model.tune_hyper_parameters()

        trained_model = training_model.train()
        evaluations = evaluate.evaluate_train_and_test(trained_model, training_model)
        format_evaluations = evaluate.format_train_and_test_evaluation(evaluations)
        print(format_evaluations)
        return trained_model, format_evaluations, None, None
        

    def __train_multi_models(self, model, df):
        if model.model_type == 'classification':
            df = self.__data_preprocessing(df, fill_missing_numeric_cells=True)
            ensemble = ClassificationEnsemble(train_df = df, target_column = model.target_column, create_encoding_rules=True, apply_encoding_rules=True,
                                              create_transformations=True, apply_transformations=True,
                                              sampling_strategy=model.sampling_strategy, scoring=model.metric)
            ensemble.create_models(df)
            # ensemble.train_all_models()
            ensemble.sort_models_by_score()
            ensemble.create_voting_classifier()
            if model.training_strategy == 'ensembleModelsTuned':
                ensemble.tuning_top_models()
            ensemble.train_voting_classifier()
            ensemble.evaluate_voting_classifier()

            evaluate = self.classificationEvaluate
            format_evaluations = evaluate.format_train_and_test_evaluation(ensemble.voting_classifier_evaluations)
            print(format_evaluations)
            return ensemble.trained_voting_classifier, format_evaluations, ensemble.encoding_rules, ensemble.transformations
        
        if model.model_type == 'regression':
            df = self.__data_preprocessing(df)
            ensemble = RegressionEnsemble(train_df = df, target_column = model.target_column, create_encoding_rules=True,
                                          apply_encoding_rules=True, create_transformations=True, apply_transformations=True, scoring=model.metric)
            ensemble.create_models(df)
            # ensemble.train_all_models()
            ensemble.sort_models_by_score()

            ensemble.create_voting_regressor()
            if model.training_strategy == 'ensembleModelsTuned':
                ensemble.tuning_top_models()
            ensemble.train_voting_regressor()
            ensemble.evaluate_voting_regressor()

            evaluate = self.regressionEvaluate
            format_evaluations = evaluate.format_train_and_test_evaluation(ensemble.voting_regressor_evaluations)
            print(format_evaluations)
            return ensemble.trained_voting_regressor, format_evaluations, ensemble.encoding_rules, ensemble.transformations
        
    def __data_preprocessing(self, df, fill_missing_numeric_cells=False):
        df_copy=df.copy()
          # df = self.data_preprocessing.one_hot_encode_all_categorical_columns(df)    
        # columns_to_encode = df.columns[df.columns != target_column]
        # df = self.data_preprocessing.fill_missing_not_numeric_cells(df)
        data_preprocessing = DataPreprocessing()
        df_copy = data_preprocessing.sanitize_dataframe(df_copy)
        if fill_missing_numeric_cells:
            df_copy = data_preprocessing.fill_missing_numeric_cells(df_copy)
        df_copy = self.data_preprocessing.convert_tdatetime_columns_to_datetime_dtype(df_copy)
        # encoding_rules = data_preprocessing.create_encoding_rules(df_copy)
        # df_copy = data_preprocessing.apply_encoding_rules(df_copy, encoding_rules)
        # df = self.data_preprocessing.one_hot_encode_column(df, 'color')
        # df = self.data_preprocessing.convert_column_categircal_values_to_numerical_values(df, 'type')
        # df = self.data_preprocessing.fill_missing_numeric_cells(df)
        # df = self.data_preprocessing.sanitize_column_names(df)
        return df_copy #, encoding_rules 


    
