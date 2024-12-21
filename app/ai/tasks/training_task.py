import traceback
from app.ai.models.classification.implementations.lightgbm_classifier import LightgbmClassifier
from app.ai.data_preprocessing import DataPreprocessing 
from app.ai.models.classification.evaluate import Evaluate as ClassificationEvaluate
from app.ai.models.regression.evaluate import Evaluate as RegressionEvaluate
from app.ai.models.regression.implementations.lightgbm_regerssor import LightGBMRegressor
from app.ai.models.regression.ensemble.ensemble import Ensemble as RegressionEnsemble
from app.ai.models.classification.ensemble.ensemble import Ensemble as ClassificationEnsemble
from app.ai.pipelines.training_pipeline import TrainingPipeline
from app.ai.tasks.llm_task import LlmTask

class TrainingTask:
    def __init__(self) -> None:
        self.classificationEvaluate = ClassificationEvaluate()
        self.regressionEvaluate = RegressionEvaluate()
        self.data_preprocessing = DataPreprocessing()
        self.llm_task = LlmTask()

    def run_task(self, model, df):
        # model.semantic_columns = [k for k, v in model.columns_type.items() if v=='semantic']

        try:    
            pipeline = TrainingPipeline()
            model.X_train, model.X_test, model.y_train, model.y_test, model.embedding_rules, model.encoding_rules, model.transformations = \
                pipeline.run_pre_training_data_pipeline(model, df)
            train_func = self.__train_multi_models if model.training_strategy in ['ensembleModelsFast', 'ensembleModelsTuned'] else self.__train_single_model
            trained_model = train_func(model)
            
                
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
            training_model.tune_hyper_parameters(model.X_train, model.y_train)

        trained_model = training_model.train(model.X_train, model.y_train)
        self.__evaluate(trained_model, model, evaluator, metric)
        return trained_model
        
    def __evaluate(self, trained_model, model, evaluator, metric):
        model.evaluations = evaluator.evaluate_train_and_test(trained_model, model.X_train, model.y_train, model.X_test, model.y_test)
        model.train_score = model.evaluations['train_metrics'][metric]
        model.test_score = model.evaluations['test_metrics'][metric]
        # evaluations = {'formated_evaluations': formated_evaluations, 'train_score': train_score, 'test_score': test_score}
        model.formated_evaluations = evaluator.format_train_and_test_evaluation(model.evaluations)
        print(model.formated_evaluations)

    def __train_multi_models(self, model):
        if model.model_type == 'classification':
            ensemble = ClassificationEnsemble(target_column = model.target_column, scoring=model.metric)
            ensemble.create_models()
            ensemble.sort_models_by_score(model.X_train, model.y_train)
            ensemble.create_voting_classifier()
            if model.training_strategy == 'ensembleModelsTuned':
                ensemble.tuning_top_models(model.X_train, model.y_train, model.X_test, model.y_test)
            ensemble.train_voting_classifier(model.X_train, model.y_train)

            evaluator = self.classificationEvaluate
            self.__evaluate(ensemble.trained_voting_classifier, model, evaluator, model.metric)
            
            return ensemble.trained_voting_classifier
        
        if model.model_type == 'regression':
            try:

                ensemble = RegressionEnsemble(target_column = model.target_column, scoring=model.metric)
                ensemble.create_models()
                ensemble.sort_models_by_score(model.X_train, model.y_train)

                ensemble.create_voting_regressor()
                if model.training_strategy == 'ensembleModelsTuned':
                    ensemble.tuning_top_models(model.X_train, model.y_train, model.X_test, model.y_test)
                ensemble.train_voting_regressor(model.X_train, model.y_train)

                evaluator = self.regressionEvaluate
                self.__evaluate(ensemble.trained_voting_regressor, model, evaluator, evaluator.get_metric_mapping(model.metric))
                
                return ensemble.trained_voting_regressor
                    
            except Exception as e:
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
                return ensemble.trained_voting_regressor
