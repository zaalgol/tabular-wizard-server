from app.ai.models.regression.ensemble.ensemble import Ensemble
from sklearn.model_selection import cross_val_score
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import OrderedDict


class EnsembleParallel(Ensemble):
    def sort_models_by_score(self):
        """Sort the regressor models by their cross-validation score using parallel processing."""
        scores = {}
        futures = {}

        # Use a ProcessPoolExecutor to parallelize cross-validation scoring
        with ProcessPoolExecutor() as executor:
            future_to_name = {
                executor.submit(self.evaluate_pretrained_model, value['model'].estimator, self.X_train, self.y_train, 5, scoring=self.scoring): name
                for name, value in self.regressors.items()
            }
            # for name, value in self.regressors.items():
            #     futures[executor.submit(self.evaluate_pretrained_model, value['model'].estimator, self.X_train, self.
            #     y_train, 5, scoring=self.scoring)] = name

            # Collect results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    scores[name] = future.result()
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")
                    scores[name] = float('-inf')  # Assign a low score to indicate failure

        # Sort regressors by average score
        sorted_names = sorted(scores, key=scores.get, reverse=True)
        self.regressors = OrderedDict((name, self.regressors[name]) for name in sorted_names)

        # Print the scores
        for name, avg_score in scores.items():
            print(f"{name}: Average CV Score = {avg_score}")

    def tuning_top_models(self):
        """Tune and retrain the top regressor models using parallel processing."""
        top_models = list(islice(self.regressors.items(), self.top_n_best_models))
        futures = {}

        # Use a ProcessPoolExecutor to parallelize tuning and training
        with ProcessPoolExecutor() as executor:
            for name, model_info in top_models:
                futures[executor.submit(self.tune_and_train, model_info, self.evaluate)] = name
                print(f"Tuning and retraining {name}...")

            # Collect the results as they complete
            for future in as_completed(futures):
                name = futures[future]
                try:
                    trained_model, evaluations = future.result()
                    self.regressors[name]['trained_model'] = trained_model
                    self.regressors[name]['evaluations'] = evaluations
                except Exception as e:
                    print(f"Error tuning and training {name}: {e}")
                    # Handle failure (e.g., by setting default evaluations or removing the model)
                        
    def evaluate_pretrained_model(self, model, X_train, y_train, cv, scoring):
        """Helper function to evaluate a model using cross-validation."""
        return cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
    
    
    def tune_and_train(self, model_info, evaluator):
        """Helper function to tune, train, and evaluate a model."""
        model_info['model'].tune_hyper_parameters(n_iter=150)
        trained_model = model_info['model'].train()
        evaluations = evaluator.evaluate_train_and_test(trained_model, model_info['model'])
        return trained_model, evaluations
