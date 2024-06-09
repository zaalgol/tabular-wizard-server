# https://chat.openai.com/c/00f0f9f4-1e06-471d-a202-cdbd2d9a6a8c
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score, log_loss, precision_score, recall_score, roc_auc_score, f1_score

class Evaluate:
    def predict(self, model, X_data):
        return model.predict(X_data)
    
    def predict_proba(self, model, X_data):
        return model.predict_proba(X_data)
    
    def get_confution_matrix (self, y_true, y_predict):
        return confusion_matrix (y_true, y_predict)

    def get_accurecy_score(self, y_true, y_predict):
        return round(accuracy_score(y_true, y_predict), 4)
    
    def get_scoring_metric(self, scoring, is_multi_class=False, is_train_multi_class=False):
        if scoring == 'accuracy':
            return 'accuracy'
        elif scoring == 'precision':
            return 'precision_macro'
        elif scoring == 'recall':
            return 'recall_macro'
        elif scoring == 'f1':
            return 'f1_macro'
        elif scoring == 'log_loss':
            return 'neg_log_loss'
        elif scoring == 'roc_auc' and is_multi_class: # and is_train_multi_class:
            return 'roc_auc_ovo'
        else:
            return scoring
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
    # Determine the number of unique classes
        num_classes = np.unique(y_true).size
        is_multi_class = num_classes > 2

        # Calculate metrics using the scoring function
        accuracy = accuracy_score(y_true, y_pred)
        loss = log_loss(y_true, y_proba)

        if is_multi_class:
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            auc = roc_auc_score(y_true, y_proba, multi_class='ovo')
        else:
            # For binary classification, set pos_label based on unique classes in y_true
            unique_classes = np.unique(y_true)
            pos_label = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            precision = precision_score(y_true, y_pred, pos_label=pos_label)
            recall = recall_score(y_true, y_pred, pos_label=pos_label)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label)
            auc = roc_auc_score(y_true, y_proba[:, 1])

        return {'accuracy': accuracy, 'log_loss': loss, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': auc}


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
        train_metrics_formatted = "\n".join([f"{key}: {value}" for key, value in evaluations['train_metrics'].items()])
        test_metrics_formatted = "\n".join([f"{key}: {value}" for key, value in evaluations['test_metrics'].items()])
        return "\n".join([
            "\nTrain eval:\n {}\n", 
            "Train metrics: \n{}\n", 
            "{}", 
            "\nTest eval:\n {}\n", 
            "Test metrics: \n{}", 
            "\n"
        ]).format(str(evaluations['train_confution_matrix']), train_metrics_formatted, "*" * 50, str(evaluations['test_confution_matrix']), test_metrics_formatted)    