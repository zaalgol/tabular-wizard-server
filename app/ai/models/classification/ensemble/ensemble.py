# https://chat.openai.com/c/60d698a6-9405-4db9-8831-6b2491bb9111
from typing import OrderedDict
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import VotingClassifier
from itertools import islice


from app.ai.models.classification.implementations.QuadraticDiscriminantAnalysisClassifier import QuadraticDiscriminantAnalysisClassifier
from app.ai.models.classification.implementations.LinearDiscriminantAnalysis_Classifier import LinearDiscriminantAnalysisClassifier
from app.ai.models.classification.implementations.NuSVC_Classifier import NuSVCClassifier
from app.ai.models.classification.implementations.base_classifier_model import BaseClassfierModel
from app.ai.models.classification.implementations.decsision_tree_classifier import DecisionTreeClassifierWrapper
from app.ai.models.classification.implementations.knn_classifier import KnnClassifier
from app.ai.models.classification.implementations.logistic_regression import LRegression
from app.ai.models.classification.implementations.mlpclassifier import MLPNetClassifier
from app.ai.models.classification.implementations.lightgbm_classifier import LightgbmClassifier
from app.ai.models.classification.implementations.random_forest_classifier import RandomForestClassifierCustom
from app.ai.models.classification.implementations.xgboost_classifier import XgboostClassifier
from app.ai.models.classification.implementations.catboot_classifier import CatboostClassifier
from app.ai.models.classification.implementations.gaussianNB_classifier import GaussianNaiveBayesClassifier
from app.ai.models.classification.implementations.BernoulliNB_classifier import BernoulliNaiveBayesClassifier
from app.ai.models.classification.implementations.svm_classifier import SvmClassifier
from app.ai.models.classification.evaluate import Evaluate
from app.ai.data_preprocessing import DataPreprocessing
from sklearn.model_selection import cross_val_score

class Ensemble(BaseClassfierModel):
    def __init__(self, target_column, scoring, number_of_n_best_models=3):
        super().__init__(target_column=target_column, scoring=scoring)
        self.classifiers = {}
        self.temp = {}
        self.evaluate = Evaluate()
        self.number_of_n_best_models = number_of_n_best_models
        self.list_of_n_best_models = []

    def create_models(self):
        model_classes = {
            'dtc_classifier':DecisionTreeClassifierWrapper,
            # 'svr_classifier':SvmClassifier, # gets stuck
            # 'nsvc_classifier':NuSVCClassifier, # doesn't have a proba
            'lgbm_classifier' :LightgbmClassifier,
            'knn_classifier' :KnnClassifier,
            'lRegression_classifier':LRegression,
            'mlp_classifier' :MLPNetClassifier,
            'rf_classifier' :RandomForestClassifierCustom,
            'gnb_classifier':GaussianNaiveBayesClassifier,
            'bnb_classifier' :BernoulliNaiveBayesClassifier,
            # 'ldac_classifier' :LinearDiscriminantAnalysisClassifier, # can't work with category target
            'qdac_classifier' :QuadraticDiscriminantAnalysisClassifier,
            'catboost_classifier':CatboostClassifier
            # 'xgb_classifier':XgboostClassifier # can't work with category target
        }
        self.classifiers = {
            key: self.__classifier_factory(model_class)
            for key, model_class in model_classes.items()
        }

    def __classifier_factory(self, model_class):
        return {
        'model': model_class(

                target_column=self.target_column, 
                scoring=self.scoring,
                # sampling_strategy='dontOversample'
            )
        }
        
    def tune_hyper_parameters(self):
        for classifier_value in self.classifiers.values():
            classifier_value['model'].tune_hyper_parameters(scoring=self.scoring)

    # def train_all_models(self):
    #     for name, classifier_value in self.classifiers.items():
    #         print(f'Training model {name}')
    #         classifier_value['trained_model'] = classifier_value['model'].train()

    def sort_models_by_score(self, X_train, y_train):
        scores = {}
        for name, value in self.classifiers.items():
            print(f"Running cross-validation for model: {name}")
            scores[name] = cross_val_score(value['model'].estimator, X_train, y_train, cv=5, scoring=self.scoring)
        average_scores = {name: score.mean() for name, score in scores.items()}
        sorted_names = sorted(average_scores, key=average_scores.get, reverse=True)
        self.classifiers = OrderedDict((name, self.classifiers[name]) for name in sorted_names)

        for name, avg_score in average_scores.items():
            print(f"{name}: Average CV Score = {avg_score}")
    
    def tuning_top_models(self):
        top_models = list(islice(self.classifiers.items(), self.number_of_n_best_models))
        for name, model_info in top_models:
            print(f"Tuning and retraining {name}...")
            model_info['model'].tune_hyper_parameters(n_iter=200)
            # model_info['trained_model'] = model_info['model'].train()
            # model_info['evaluations'] = self.evaluate.evaluate_train_and_test(model_info['trained_model'], model_info['model'])
            # self.temp[name]=model_info['evaluations']

    def create_voting_classifier(self):
        self.list_of_n_best_models = [(name, info['model'].estimator) for name, info in islice(self.classifiers.items(), self.number_of_n_best_models)]
        self.voting_classifier = VotingClassifier(estimators=self.list_of_n_best_models, voting='soft')

    def train_voting_classifier(self, X_train, y_train):
        self.trained_voting_classifier = self.voting_classifier.fit(X_train, y_train)

    # def evaluate_voting_classifier(self, X_train, y_train, X_test, y_test):
    #     self.evaluations = self.evaluate.evaluate_train_and_test(self.trained_voting_classifier, X_train, y_train, X_test, y_test)
