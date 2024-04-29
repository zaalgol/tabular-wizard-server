from abc import abstractmethod
import pprint
import time
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from skopt.callbacks import DeadlineStopper, DeltaYStopper

from src.data_preprocessing import DataPreprocessing


class BaseModel:
    def __init__(self, train_df, target_column, scoring, split_column, 
                 create_encoding_rules=False, apply_encoding_rules=False, create_transformations=False, apply_transformations=False, test_size=0.2,
                 already_splitted_data=None):
        self.search = None
        self.scoring = scoring
        self.target_column = target_column
        self.data_preprocessing = DataPreprocessing()
        self.encoding_rules = None


        if already_splitted_data:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                already_splitted_data['X_train'], already_splitted_data['X_test'], already_splitted_data['y_train'], already_splitted_data['y_test']
            return

        if split_column is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_df,
                                                                                        train_df[target_column],
                                                                                        test_size=test_size, random_state=42)
        else:
            splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7)
            split = splitter.split(train_df, groups=train_df[split_column])
            train_inds, test_inds = next(split)

            train = train_df.iloc[train_inds]
            self.y_train = train[[target_column]].astype(float)
            test = train_df.iloc[test_inds]
            self.y_test = test[[target_column]].astype(float)

        self.X_train = self.X_train.drop([target_column], axis=1)
        self.X_test = self.X_test.drop([target_column], axis=1)

        if create_encoding_rules:
            self.encoding_rules = self.data_preprocessing.create_encoding_rules(self.X_train)
        if apply_encoding_rules:
            self.X_train = self.data_preprocessing.apply_encoding_rules(self.X_train, self.encoding_rules)
            self.X_test = self.data_preprocessing.apply_encoding_rules(self.X_test, self.encoding_rules)

        if create_transformations:
            numeric_columns = self.data_preprocessing.get_numeric_columns(self.X_train)
            self.transformations = self.data_preprocessing.create_transformed_numeric_column_details(self.X_train, numeric_columns)
        if apply_transformations:
            self.X_train = self.data_preprocessing.transformed_numeric_column_details(self.X_train, self.transformations)
            self.X_test = self.data_preprocessing.transformed_numeric_column_details(self.X_test, self.transformations)


    @property
    def callbacks(self):
        time_limit_control = DeadlineStopper(total_time=60 * 45) # We impose a time limit (45 minutes]
        return [time_limit_control]
    
        # TODO: make it work. corrently, it stoppes very early.
        # overdone_control = DeltaYStopper(delta=0.0001) # We stop if the gain of the optimization becomes too small
        # return [overdone_control, time_limit_control]

    @property
    @abstractmethod
    def default_params(self):
        return {}
    
    