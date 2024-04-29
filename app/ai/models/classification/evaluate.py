# https://chat.openai.com/c/00f0f9f4-1e06-471d-a202-cdbd2d9a6a8c
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, roc_auc_score, f1_score

class Evaluate:
    def predict(self, model, X_data):
        return model.predict(X_data)
    
    def predict_proba(self, model, X_data):
        return model.predict_proba(X_data)
    
    def get_confution_matrix (self, y_true, y_predict):
        return confusion_matrix (y_true, y_predict)

    def get_accurecy_score(self, y_true, y_predict):
        return round(accuracy_score(y_true, y_predict), 4)
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        # Determine the number of unique classes
        num_classes = np.unique(y_true).size

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        loss = log_loss(y_true, y_proba)

        if num_classes == 2:
            # Binary classification metrics
            accuracy = self.get_accurecy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)  # Direct calculation for binary
            auc = roc_auc_score(y_true, y_proba[:, 1])  # Use probabilities for the positive class
        else:
            # Multi-class classification metrics
            accuracy = self.get_accurecy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')  # Specify averaging method
            auc = roc_auc_score(y_true, y_proba, multi_class='ovo')  # Use One-vs-One strategy

        return {'accuracy':accuracy, 'log loss': loss, 'precision': precision, 'recall': recall, 'f1': f1, 'roc auc': auc}

    def evaluate_train_and_test(self, model, classifier):
        y_predict = model.predict(classifier.X_train)
        y_probability = model.predict_proba(classifier.X_train)
        # train_score = self.get_accurecy_score(classifier.y_train, y_predict)
        train_metrics = self.calculate_metrics(classifier.y_train, y_predict, y_probability)
        train_confution_matrix = self.get_confution_matrix(classifier.y_train, y_predict)
        train_confution_matrix_str = "\n".join([classification_report(classifier.y_train, y_predict), f"confusion_matrix: \n {train_confution_matrix}"])

        y_predict = model.predict(classifier.X_test)
        y_probability = model.predict_proba(classifier.X_test)
        test_metrics = self.calculate_metrics(classifier.y_test, y_predict, y_probability)
        test_confution_matrix = self.get_confution_matrix (classifier.y_test, y_predict)
        test_confution_matrix_str = "\n".join([classification_report(classifier.y_test, y_predict), f"confusion_matrix: \n {test_confution_matrix}"])

        # return "\n".join(["\nTrain eval:",  str(train_confution_matrix ), f'score: {train_score}',  "\nTest eval:", str(test_confution_matrix ), f'score: {test_score}', "*" * 100, "\n"])
        return {'train_confution_matrix': train_confution_matrix_str, 'train_metrics':train_metrics,
                 'test_confution_matrix':test_confution_matrix_str, 'test_metrics': test_metrics}
    
    def format_train_and_test_evaluation(self, evaluations):
         return "\n".join([
            "\nTrain eval:\n {}", 
            "Train metrics: {}", 
            "{}", 
            "\nTest eval:\n {}", 
            "Test metrics: {}", 
            "\n"
        ]).format(str(evaluations['train_confution_matrix']), evaluations['train_metrics'], "*" * 100, str(evaluations['test_confution_matrix']), evaluations['test_metrics'])
    