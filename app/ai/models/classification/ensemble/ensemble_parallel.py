from app.ai.models.classification.ensemble.ensemble import Ensemble
from sklearn.model_selection import cross_val_score
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import OrderedDict


class EnsembleParallel(Ensemble):
    def sort_models_by_score(self):
        """Sorts models by their cross-validation score using parallel processing."""
        scores = {}
        
        # Use a ProcessPoolExecutor to parallelize cross-validation scoring
        with ProcessPoolExecutor() as executor:
            # Submit evaluation tasks for each model
            future_to_name = {
                executor.submit(self.evaluate_pretrained_model, value['model'].estimator, self.X_train, self.y_train, 5): name
                for name, value in self.classifiers.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    scores[name] = future.result()
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")
                    scores[name] = float('-inf')  # Assign a low score to indicate failure

        # Sort classifiers by average score
        sorted_names = sorted(scores, key=scores.get, reverse=self.scoring != 'log loss')
        self.classifiers = OrderedDict((name, self.classifiers[name]) for name in sorted_names)

        # Print the scores
        for name, avg_score in scores.items():
            print(f"{name}: Average CV Score = {avg_score}")
            

    def tuning_top_models(self):
        """Tune and retrain the top models using parallel processing."""
        top_models = list(islice(self.classifiers.items(), self.top_n_best_models))
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
                    self.classifiers[name]['trained_model'] = trained_model
                    self.classifiers[name]['evaluations'] = evaluations
                    self.temp[name] = evaluations
                except Exception as e:
                    print(f"Error tuning and training {name}: {e}")
                    # Handle failure if needed (e.g., by setting default evaluations or removing the model)
                    
    def evaluate_pretrained_model(self, model, X_train, y_train, cv):
        """Helper function to evaluate a model using cross-validation."""
        return cross_val_score(model, X_train, y_train, cv=cv).mean()
    
    def tune_and_train(self, model_info, evaluator):
        """Helper function to tune, train, and evaluate a model."""
        model_info['model'].tune_hyper_parameters(n_iter=150)
        trained_model = model_info['model'].train()
        evaluations = evaluator.evaluate_train_and_test(trained_model, model_info['model'])
        return trained_model, evaluations
