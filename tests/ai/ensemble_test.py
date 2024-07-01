from conftest import app, Ensemble, BaseClassfierModel

import pandas as pd


# import pytest

class TestEnsemble:

    # create_models initializes all classifiers correctly
    def test_create_models_initializes_classifiers(self):
        # Sample data
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        }
        import pandas as pd

        df = pd.DataFrame(data)

        # Initialize the Ensemble class
        ensemble = Ensemble(train_df=df, target_column='target', 
                            split_column=None, create_encoding_rules=False, apply_encoding_rules=False,
                            create_transformations=False, apply_transformations=False, scoring='accuracy', 
                            sampling_strategy='conditionalOversampling', number_of_n_best_models=3)

        # Create models
        ensemble.create_models(df)

        # Check if all classifiers are initialized
        expected_classifiers = [
            'dtc_classifier', 'svr_classifier', 'lgbm_classifier', 'knn_classifier', 
            'lRegression_classifier', 'mlp_classifier', 'rf_classifier', 'gnb_classifier', 
            'bnb_classifier', 'ldac_classifier', 'qdac_classifier', 'catboost_classifier'
        ]

        for classifier in expected_classifiers:
            assert classifier in ensemble.classifiers
            assert isinstance(ensemble.classifiers[classifier]['model'], BaseClassfierModel)
