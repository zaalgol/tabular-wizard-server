from app.ai.models.classification.implementations.lightgbm_classifier import LightgbmClassifier
from app.ai.data_preprocessing import DataPreprocessing 
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
from app.ai.models.regression.implementations.lightgbm_regerssor import LightGBMRegressor
from app.ai.models.regression.ensemble.ensemble import Ensemble as RegressionEnsemble
from app.ai.models.classification.ensemble.ensemble import Ensemble as ClassificationEnsemble


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
                trained_model, evaluations, encoding_rules, transformations = self.__train_multi_models(model, df.copy())
            else:
                trained_model, evaluations, encoding_rules, transformations = self.__train_single_model(model, df.copy())
            is_training_successfully_finished = True
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        finally:
            model.formated_evaluations = evaluations['formated_evaluations']
            model.train_score = evaluations['train_score']
            model.test_score = evaluations['test_score']
            task_callback(df, model, trained_model, encoding_rules, transformations,  headers, is_training_successfully_finished, app_context)

    def __train_single_model(self, model, df):
        df = self.__data_preprocessing(df, fill_missing_numeric_cells=True)
        metric = model.metric
        if model.model_type == 'classification':
            training_model = LightgbmClassifier(train_df = df, target_column = model.target_column, scoring=model.metric, 
                                                sampling_strategy=model.sampling_strategy)
            evaluate = self.classificationEvaluate

        elif model.model_type == 'regression':
            training_model = LightGBMRegressor(train_df = df, target_column = model.target_column, scoring=model.metric)
            evaluate = self.regressionEvaluate
            metric = evaluate.get_metric_mapping(model.metric)

        if model.training_strategy == 'singleModelTuned':
            training_model.tune_hyper_parameters()

        trained_model = training_model.train()
        evaluations = evaluate.evaluate_train_and_test(trained_model, training_model)
        formated_evaluations = evaluate.format_train_and_test_evaluation(evaluations)
        print(formated_evaluations)

        train_score = evaluations['train_metrics'][metric]
        test_score = evaluations['test_metrics'][metric]
        evaluations = {'formated_evaluations': formated_evaluations, 'train_score': train_score, 'test_score': test_score}
        
        return trained_model, evaluations, None, None
        

    def __train_multi_models(self, model, df):
        if model.model_type == 'classification':
            df = self.__data_preprocessing(df, fill_missing_numeric_cells=True)
            ensemble = ClassificationEnsemble(train_df = df, target_column = model.target_column, create_encoding_rules=True, apply_encoding_rules=True,
                                              create_transformations=True, apply_transformations=True,
                                              sampling_strategy=model.sampling_strategy, scoring=model.metric)
            ensemble.create_models(df)
            ensemble.sort_models_by_score()
            ensemble.create_voting_classifier()
            if model.training_strategy == 'ensembleModelsTuned':
                ensemble.tuning_top_models()
            ensemble.train_voting_classifier()
            ensemble.evaluate_voting_classifier()

            evaluate = self.classificationEvaluate
            formated_evaluations = evaluate.format_train_and_test_evaluation(ensemble.voting_classifier_evaluations)
            print(formated_evaluations)
            train_score = ensemble.voting_classifier_evaluations['train_metrics'][model.metric]
            test_score = ensemble.voting_classifier_evaluations['test_metrics'][model.metric]
            evaluations = {'formated_evaluations': formated_evaluations, 'train_score': train_score, 'test_score': test_score}
            
            return ensemble.trained_voting_classifier, evaluations, ensemble.encoding_rules, ensemble.transformations
        
        if model.model_type == 'regression':
            try:
                df = self.__data_preprocessing(df)
                ensemble = RegressionEnsemble(train_df = df, target_column = model.target_column, create_encoding_rules=True,
                                            apply_encoding_rules=True, create_transformations=True, apply_transformations=True, scoring=model.metric)
                ensemble.create_models(df)
                ensemble.sort_models_by_score()

                ensemble.create_voting_regressor()
                if model.training_strategy == 'ensembleModelsTuned':
                    ensemble.tuning_top_models()
                ensemble.train_voting_regressor()
                ensemble.evaluate_voting_regressor()

                evaluate = self.regressionEvaluate
                formated_evaluations = evaluate.format_train_and_test_evaluation(ensemble.voting_regressor_evaluations)
                print(formated_evaluations)
                metric = self.regressionEvaluate.get_metric_mapping(model.metric)
                train_score = ensemble.voting_regressor_evaluations['train_metrics'][metric]
                test_score = ensemble.voting_regressor_evaluations['test_metrics'][metric]
                evaluations = {'formated_evaluations': formated_evaluations, 'train_score': train_score, 'test_score': test_score}
                
            except Exception as e:
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
            return ensemble.trained_voting_regressor, evaluations, ensemble.encoding_rules, ensemble.transformations
            
        
    def __data_preprocessing(self, df, fill_missing_numeric_cells=False):
        df_copy=df.copy()
        data_preprocessing = DataPreprocessing()
        df_copy = data_preprocessing.sanitize_dataframe(df_copy)
        if fill_missing_numeric_cells:
            df_copy = data_preprocessing.fill_missing_numeric_cells(df_copy)
        df_copy = self.data_preprocessing.convert_tdatetime_columns_to_datetime_dtype(df_copy)
        return df_copy 


    
