import traceback
from app.ai.models.classification.implementations.lightgbm_classifier import LightgbmClassifier
from app.ai.data_preprocessing import DataPreprocessing 
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
from app.ai.models.regression.implementations.lightgbm_regerssor import LightGBMRegressor
from app.ai.models.regression.ensemble.ensemble import Ensemble as RegressionEnsemble
from app.ai.models.classification.ensemble.ensemble import Ensemble as ClassificationEnsemble
from app.ai.pipeline import Pipeline
from app.tasks.llm_task import LlmTask

class TrainingTask:
    def __init__(self) -> None:
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.data_preprocessing = DataPreprocessing()
        self.llm_task = LlmTask()

    def run_task(self, model, df):
        # model.semantic_columns = [k for k, v in model.columns_type.items() if v=='semantic']

        try:    
            pipeline = Pipeline()
            model.X_train, model.X_test, model.y_train, model.y_test, model.embedding_rules, model.encoding_rules, model.transformations = \
                pipeline.run_pre_training_data_pipeline(model, df)
            train_func = self.__train_multi_models if model.training_strategy in ['ensembleModelsFast', 'ensembleModelsTuned'] else self.__train_single_model
            trained_model = train_func(model)
            

        # model.train_score = evaluations['train_metrics'][metric]
        # model.test_score = evaluations['test_metrics'][metric]
        # model.formated_evaluations = {'formated_evaluations': formated_evaluations, 'train_score': model.train_score, 'test_score': model.test_score}
        
            # # return trained_model, model.embedding_rules, None, None
            # model.train_score = evaluations['train_metrics'][model.metric]
            # model.test_score = evaluations['test_metrics'][model.metric]
            # # evaluations = {'formated_evaluations': formated_evaluations, 'train_score': train_score, 'test_score': test_score}
            # model.evaluations = evaluations
            # model.formated_evaluations = formated_evaluations
            # model.formated_evaluations
                
            return trained_model, True
            
        except Exception as e:
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
            return (None,) * 8

    def __train_single_model(self, model):
        # df = self.__data_preprocessing(df, model, fill_missing_numeric_cells=True)
        metric = model.metric
        if model.model_type == 'classification':
            training_model = LightgbmClassifier(target_column = model.target_column, scoring=model.metric)
            evaluator = self.classificationEvaluate

        elif model.model_type == 'regression':
            training_model = LightGBMRegressor(target_column = model.target_column, scoring=model.metric)
            evaluator = self.regressionEvaluate
            metric = evaluator.get_metric_mapping(model.metric)

        if model.training_strategy == 'singleModelTuned':
            training_model.tune_hyper_parameters()

        trained_model = training_model.train(model.X_train, model.y_train)
        self.__evaluate(trained_model, model, evaluator, metric)
        return trained_model
        # evaluations = evaluator.evaluate_train_and_test(trained_model, model.X_train, model.y_train, model.X_test, model.y_test)
        # formated_evaluations = evaluator.format_train_and_test_evaluation(evaluations)
        # print(formated_evaluations)

        # return trained_model, evaluations, formated_evaluations

        # model.train_score = evaluations['train_metrics'][metric]
        # model.test_score = evaluations['test_metrics'][metric]
        # model.formated_evaluations = {'formated_evaluations': formated_evaluations, 'train_score': model.train_score, 'test_score': model.test_score}
        
        # return trained_model, model.embedding_rules, None, None
        # train_score = evaluations['train_metrics'][metric]
        # test_score = evaluations['test_metrics'][metric]
        # evaluations = {'formated_evaluations': formated_evaluations, 'train_score': train_score, 'test_score': test_score}
        # model.evaluations = evaluations
        # return trained_model
        
    def __evaluate(self, trained_model, model, evaluator, metric):
        model.evaluations = evaluator.evaluate_train_and_test(trained_model, model.X_train, model.y_train, model.X_test, model.y_test)
        model.train_score = model.evaluations['train_metrics'][metric]
        model.test_score = model.evaluations['test_metrics'][metric]
        # evaluations = {'formated_evaluations': formated_evaluations, 'train_score': train_score, 'test_score': test_score}
        model.formated_evaluations = evaluator.format_train_and_test_evaluation(model.evaluations)
        print(model.formated_evaluations)

    def __train_multi_models(self, model, df):
        if model.model_type == 'classification':
            # df = self.__data_preprocessing(df, model, fill_missing_numeric_cells=True)
            ensemble = ClassificationEnsemble(train_df = df, target_column = model.target_column, scoring=model.metric)
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
            model.train_score = ensemble.voting_classifier_evaluations['train_metrics'][model.metric]
            model.test_score = ensemble.voting_classifier_evaluations['test_metrics'][model.metric]
            model.formated_evaluations = {'formated_evaluations': formated_evaluations, 'train_score': model.train_score, 'test_score': model.test_score}
            
            return ensemble.trained_voting_classifier, ensemble.embedding_rules, ensemble.encoding_rules, ensemble.transformations
        
        if model.model_type == 'regression':
            try:
                # df = self.__data_preprocessing(df, model, fill_missing_numeric_cells=True)
                ensemble = RegressionEnsemble(train_df = df, target_column = model.target_column, semantic_columns = model.semantic_columns, create_encoding_rules=True,
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
                model.train_score = ensemble.voting_regressor_evaluations['train_metrics'][metric]
                model.test_score = ensemble.voting_regressor_evaluations['test_metrics'][metric]
                model.evaluations = {'formated_evaluations': formated_evaluations, 'train_score': model.train_score, 'test_score': model.test_score}
                
            except Exception as e:
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
            return ensemble.trained_voting_regressor, ensemble.embedding_rules, ensemble.encoding_rules, ensemble.transformations
            
        
    # def __data_preprocessing(self, df, model, fill_missing_numeric_cells=False):
    #     df_copy=df.copy()
    #     if model.is_time_series:
    #         df_copy, model.time_series_code = self.llm_task.use_llm_toproccess_timeseries_dataset(df_copy, model.target_column)
    #     data_preprocessing = DataPreprocessing()
    #     # df_copy = data_preprocessing.sanitize_dataframe(df_copy)
    #     if fill_missing_numeric_cells:
    #         df_copy = data_preprocessing.fill_missing_numeric_cells(df_copy)
    #     df_copy = self.data_preprocessing.convert_datetime_columns_to_datetime_dtype(df_copy, model)
    #     return df_copy 


    
