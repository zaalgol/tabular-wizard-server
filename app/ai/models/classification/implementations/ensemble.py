# https://chat.openai.com/c/60d698a6-9405-4db9-8831-6b2491bb9111
from typing import OrderedDict
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import VotingClassifier
from itertools import islice


from src.models.classification.implementations.base_classifier_model import BaseClassfierModel
from src.models.classification.implementations.knn_classifier import KnnClassifier
from src.models.classification.implementations.logistic_regression import LRegression
from src.models.classification.implementations.mlpclassifier import MLPNetClassifier
from src.models.classification.implementations.lightgbm_classifier import LightgbmClassifier
from src.models.classification.implementations.random_forest_classifier import RandomForestClassifierCustom
from src.models.classification.implementations.xgboost_classifier import XgboostClassifier
from src.models.classification.implementations.catboot_classifier import CatboostClassifier
from src.models.classification.implementations.naive_bayes_classifier import NaiveBayesClassifier
from src.models.classification.implementations.svm_classifier import SvmClassifier
from src.models.classification.evaluate import Evaluate
from src.data_preprocessing import DataPreprocessing
from sklearn.model_selection import cross_val_score

class Ensemble(BaseClassfierModel):
    def __init__(self, train_df, target_column, split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                  create_transformations=False, apply_transformations=False, test_size=0.3, scoring='accuracy', 
                  sampling_strategy='conditionalOversampling', top_n_best_models=3):
        super().__init__(train_df=train_df, target_column=target_column, scoring=scoring, split_column=split_column, test_size=test_size,
                    create_encoding_rules=create_encoding_rules, apply_encoding_rules=apply_encoding_rules, 
                    create_transformations=create_transformations, apply_transformations=apply_transformations, sampling_strategy=sampling_strategy)
        self.classifiers = {}
        self.temp = {}
        self.already_splitted_data = {'X_train': self.X_train, 'X_test': self.X_test, 'y_train': self.y_train, 'y_test':self.y_test}
        self.evaluate = Evaluate()
        self.top_n_best_models = top_n_best_models

    def create_models(self, df):
        self.classifiers['svr_classifier'] = {'model':SvmClassifier(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        self.classifiers['cat_classifier'] = {'model':CatboostClassifier(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        self.classifiers['lgbm_classifier'] = {'model':LightgbmClassifier(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        self.classifiers['knn_classifier'] = {'model':KnnClassifier(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        self.classifiers['LRegression'] = {'model':LRegression(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        self.classifiers['mlp_classifier'] = {'model':MLPNetClassifier(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        self.classifiers['rf_classifier'] = {'model':RandomForestClassifierCustom(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        self.classifiers['nb_classifier'] = {'model':NaiveBayesClassifier(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}
        if df[self.target_column].dtype not in ['category', 'object']:
            self.classifiers['xgb_classifier'] = {'model':XgboostClassifier(train_df = df.copy(), target_column = self.target_column, already_splitted_data=self.already_splitted_data, sampling_strategy='dontOversample')}

        
    def tune_hyper_parameters(self):
        for classifier_value in self.classifiers.values():
            classifier_value['model'].tune_hyper_parameters(scoring=self.scoring)

    def train_all_models(self):
        for classifier_value in self.classifiers.values():
            classifier_value['trained_model'] = classifier_value['model'].train()

    def sort_models_by_score(self):
        scores = {name: cross_val_score(value['model'].estimator, self.X_train, self.y_train, cv=5) for name, value in self.classifiers.items()}
        average_scores = {name: score.mean() for name, score in scores.items()}
        sorted_names = sorted(average_scores, key=average_scores.get, reverse=self.scoring != 'log loss')
        self.classifiers = OrderedDict((name, self.classifiers[name]) for name in sorted_names)

        for name, avg_score in average_scores.items():
            print(f"{name}: Average CV Score = {avg_score}")
        # for classifier_value in self.classifiers.values():
        #     classifier_value['evaluations'] = self.evaluate.evaluate_train_and_test(classifier_value['trained_model'], classifier_value['model'])
        # self.classifiers= dict(sorted(self.classifiers.items(), key=lambda item:
        #     item[1]['evaluations']['test_metrics'][self.scoring], reverse=self.scoring != 'loss')) # for loss metrics, lower is better, so we will sort in ascending order


    def tuning_top_models(self):
        top_models = list(islice(self.classifiers.items(), self.top_n_best_models))
        for name, model_info in top_models:
            print(f"Tuning and retraining {name}...")
            model_info['model'].tune_hyper_parameters(n_iter=150)
            model_info['trained_model'] = model_info['model'].train()
            model_info['evaluations'] = self.evaluate.evaluate_train_and_test(model_info['trained_model'], model_info['model'])
            self.temp[name]=model_info['evaluations']

    def create_voting_classifier(self):
        model_list = [(name, info['model'].estimator) for name, info in islice(self.classifiers.items(), self.top_n_best_models)]
        self.voting_classifier = VotingClassifier(estimators=model_list, voting='soft')

    def train_voting_classifier(self):
        self.trained_voting_classifier = self.voting_classifier.fit(self.X_train, self.y_train)

    def evaluate_voting_classifier(self):
        self.voting_classifier_evaluations = self.evaluate.evaluate_train_and_test(self.trained_voting_classifier, self)

    def hard_predict(self, trained_models):
        predictions = [model.predict(X_test) for model, X_test in trained_models]

        # Use mode to find the most common class label
        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()


if __name__ == "__main__":
    # target_column = 'price_range'
    # train_path = "tabularwizard/datasets/phone-price-classification/train.csv"
    
    # target_column = 'Survived'
    # train_path = "tabularwizard/datasets/titanic.csv"
    
    target_column = 'species'
    train_path = "tabularwizard/datasets/IRIS.csv"
    
    
    
    
    # train_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/train.csv"
    train_data = pd.read_csv(train_path)
    # train_data_capy = train_data.copy()

    data_preprocessing = DataPreprocessing()
    train_data = data_preprocessing.sanitize_dataframe(train_data)
    train_data = data_preprocessing.fill_missing_numeric_cells(train_data)
    
    # train_data['FamilySize'] = train_data ['SibSp'] + train_data['Parch'] + 1
    # train_data['IsAlone'] = 1 #initialize to yes/1 is alone
    # train_data['IsAlone'].loc[train_data['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    # train_data['FareBin'] = pd.qcut(train_data['Fare'], 4)
    # train_data['AgeBin'] = pd.cut(train_data['Age'].astype(int), 5)

    # train_data = data_preprocessing.exclude_columns(train_data, [target_column])
    # train_data[target_column] = train_data_capy[target_column]
    ensemble = Ensemble(train_df=train_data, target_column=target_column,
                         create_encoding_rules=True, apply_encoding_rules=True,
                         create_transformations=True, apply_transformations=True)
    ensemble.create_models(train_data)
    # ensemble.train_all_models()
    ensemble.sort_models_by_score()
    # for name, value in ensemble.classifiers.items():
    #     print("<" * 20 +  f" Name {name}, train: {value['evaluations']['train_metrics'][ensemble.scoring]} test: {value['evaluations']['test_metrics'][ensemble.scoring]}")

    ensemble.create_voting_classifier()
    ensemble.train_voting_classifier()
    ensemble.evaluate_voting_classifier()

    # for name, value in ensemble.classifiers.items():
    #     print("<" * 20 +  f" Name {name}, train: {value['evaluations']['train_metrics'][ensemble.scoring]} test: {value['evaluations']['test_metrics'][ensemble.scoring]}")
    print(ensemble.evaluate.format_train_and_test_evaluation(ensemble.voting_classifier_evaluations))
    


    






        


    