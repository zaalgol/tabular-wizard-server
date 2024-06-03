import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error

class Evaluate:
    def __init__(self):
        self.metric_mapping = {
            'r2': 'R2',
            'neg_root_mean_squared_error': 'RMSE',
            'neg_mean_squared_error': 'MSE',
            'neg_mean_absolute_error': 'MAE',
            'neg_mean_absolute_percentage_error': 'MAPE'
        }

    def predict(self, model, X_data):
        return model.predict (X_data)
    
    def get_metric_mapping(self, metric):
        return self.metric_mapping.get(metric, metric)

    def calculate_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)  # Calculating RMSE
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        # rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

        return {'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE':rmse, 'MAPE': mape}# , 'RMSLE':rmsle }
    
    def evaluate_train_and_test(self, model, regressor):
        y_train_predict = self.predict(model, regressor.X_train)
        train_metrics = self.calculate_metrics(regressor.y_train, y_train_predict)
        # model["train_evaluation"] = train_evaluation

        y_test_predict = self.predict(model, regressor.X_test)
        test_metrics = self.calculate_metrics(regressor.y_test, y_test_predict)

        return {'train_metrics': train_metrics, 'test_metrics': test_metrics}
        # model["test_evaluation"] = test_evaluation

    
    def format_train_and_test_evaluation(self, evaluations):
         return "\n".join([
            "Train metrics: {}", 
            "Test metrics: {}", 
        ]).format(evaluations['train_metrics'], evaluations['test_metrics'])



            