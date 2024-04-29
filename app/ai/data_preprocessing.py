import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.utils import resample

class DataPreprocessing:
    def exclude_columns(self, df, columns_to_exclude):
        # Use intersection to find the columns that exist in both the dataframe and the columns_to_exclude list
        valid_columns_to_exclude = [col for col in columns_to_exclude if col in df.columns]
        # Drop only the valid columns
        return df.drop(columns=valid_columns_to_exclude).copy()
        
    def exclude_other_columns(self, df, columns):
        columns_to_keep = [col for col in columns if col in df.columns]
        return df[columns_to_keep]
    
    def exclude_non_numeric_columns_with_large_unique_values(self, df, k):
    
        cols_to_drop = [col for col in df.select_dtypes(exclude=[np.number]).columns
                    if df[col].nunique() > k]

        # Drop these columns from the DataFrame
        df.drop(columns=cols_to_drop, inplace=True)

        return df

    def sanitize_column_names(self, df):
        new_df = df.copy()
        new_df.columns = [col.replace(',', '').replace(':', '').replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '') for col in df.columns]
        return new_df
    
    def sanitize_cells(self, df):
        new_df = df.copy()
        for col in new_df.columns:
            if new_df[col].dtype == 'object' or new_df[col].dtype =='category':  # Assuming we only want to sanitize string-type columns
                new_df[col] = new_df[col].apply(lambda x: x.replace(',', '').replace(':', '').replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '') if isinstance(x, str) else x)
        return new_df
    
    def sanitize_dataframe(self, df):
        new_df = self.sanitize_column_names(df)
        return self.sanitize_cells(new_df)
    
    def convert_column_categircal_values_to_numerical_values(self, df, column):
        df_copy = df.copy()
        labels, _ = pd.factorize(df_copy[column])
        df_copy[column] = labels
        return df_copy
    
    def transfer_column(self, df, column):
        df_copy = df.copy()
        le = LabelEncoder()
        transformed_column = le.fit_transform(df[column])
        df_copy[column]=transformed_column
        return df_copy
    
    def create_transformed_numeric_column_details(self, df, numeric_columns):
        df_copy = df.copy()
        transformed_column_details = {}

        for column in numeric_columns:
        # Initialize and fit the scaler
            scaler = StandardScaler()
            scaler.fit(df_copy[[column]])
            transformed_column_details[column] = {'mean': scaler.mean_[0], 'scale': scaler.scale_[0]}

        return transformed_column_details
    
    def transformed_numeric_column_details(self, df, transformed_column_details):
        df_copy = df.copy()
        for column, details in transformed_column_details.items():
            # Retrieve the scaler document for the column

            # Recreate the scaling transformation manually
            mean = details['mean']
            scale = details['scale']
            df_copy[column] = (df_copy[column] - mean) / scale
        
        return df_copy

    # example of mapping_dict: {'high': 3, 'medium': 2, 'low': 1}
    def map_order_column(self, df, column_name, mapping_dict):
        df_copy = df.copy()
        df_copy[column_name] = df_copy[column_name].map(mapping_dict)
        return df_copy
    
    # example of mapping_dict: {'high': 3, 'medium': 2, 'low': 1}
    def reverse_map_order_column(self, df, column_name, mapping_dict):
        # Inverting the mapping_dict
        inverted_dict = {v: k for k, v in mapping_dict.items()}
        df_copy = df.copy()
        df_copy[column_name] = df_copy[column_name].map(inverted_dict)
        return df_copy
    
    def one_hot_encode_columns(self, dataframe, column_name_array):
        df = dataframe.copy()
        for column_name in column_name_array:
            df = self.one_hot_encode_column(df, column_name)
        return df
    
    
    def one_hot_encode_all_categorical_columns(self, df):
        return pd.get_dummies(df)

    def one_hot_encode_column(self, dataframe, column_name):
        # Make a copy of the original DataFrame to avoid modifying it in place
        encoded_df = dataframe.copy()
        
        # Use pd.get_dummies to perform one-hot encoding on the specified column
        one_hot_encoded = pd.get_dummies(encoded_df[column_name], prefix=column_name)
        
        # Concatenate the one-hot encoded columns to the original DataFrame and drop the original column
        encoded_df = pd.concat([encoded_df, one_hot_encoded], axis=1)
        encoded_df.drop(columns=[column_name], inplace=True)
        
        return encoded_df
    
    def set_not_numeric_as_categorial(self, df):
        df_copy = df.copy()
        cat_features  =  self.get_all_categorical_columns_names(df_copy)
        for feature in cat_features:
            df_copy[feature] = df_copy[feature].astype('category')
        return df_copy
    
    def get_all_categorical_columns_names(self, df):
        return [f for f in df.columns if df.dtypes[f] == 'object' or df.dtypes[f] == 'category']
    
    def get_numeric_columns(self, df):
        return [f for f in df.columns if df.dtypes[f] != 'object'or df.dtypes[f] == 'category']
    
    
    def fill_missing_numeric_cells(self, df, median_stratay=False):
        new_df = df.copy()
        if median_stratay:
            new_df.fillna(df.median(numeric_only=True).round(1), inplace=True)
        else:
            new_df.fillna(df.mean(numeric_only=True).round(1), inplace=True)
        return new_df
    
    def fill_missing_not_numeric_cells(self, df):
        new_df = df.copy()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
        # Filling missing values in categorical columns with their respective modes.
        for column in categorical_columns:
            mode_value = new_df[column].mode()[0]  # Getting the mode value of the column
            new_df[column].fillna(mode_value, inplace=True)
    

        return new_df
    
    def get_missing_values_per_coloun(self, df):
        return df.isnull().sum()
    
    def scale_values_netween_0_to_1(self, df, columns):
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=columns)
        return df
    
    def describe_datafranme(self, df):
        print(df.describe().T)
        return (df.describe().T)
    
    def oversampling_minority_class(self, df, column_name, minority_class, majority_class):
        new_df = df.copy()
        minority_rows = new_df[new_df[column_name]==minority_class]
        majority_rows = new_df[new_df[column_name]==majority_class]

        minority_upsampled = resample(minority_rows,
                          replace=True, # sample with replacement
                          n_samples=len(majority_rows), # match number in majority class
                          random_state=27) # reproducible results

        # combine majority and upsampled minority
        return pd.concat([majority_rows, minority_upsampled])
            
    def majority_minority_class(self, df, column_name, minority_class, majority_class):
        new_df = df.copy()
        minority_rows = new_df[new_df[column_name]==minority_class]
        majority_rows = new_df[new_df[column_name]==majority_class]

        majority_underampled = resample(majority_rows,
                          replace=False, # sample with replacement
                          n_samples=len(minority_rows), # match number in majority class
                          random_state=27) # reproducible results

        # combine majority and upsampled minority
        return pd.concat([majority_underampled, minority_rows])
    
    def oversampling_minority_classifier(self, classifier, minority_class, majority_class):
        sampling_df = self.oversampling_minority_class(pd.concat([classifier.X_train, classifier.y_train], axis=1),
                                                         classifier.y_train.name, minority_class, majority_class)
        print(sampling_df.Class.value_counts())
        classifier.y_train = sampling_df.Class
        classifier.X_train = sampling_df.drop('Class', axis=1)
        

    def majority_minority_classifier(self, classifier, minority_class, majority_class):
        sampling_df = self.majority_minority_class(pd.concat([classifier.X_train, classifier.y_train], axis=1),
                                                         classifier.y_train.name, minority_class, majority_class)
        classifier.y_train = sampling_df.Class
        classifier.X_train = sampling_df.drop('Class', axis=1)

    def create_encoding_rules(self, df, threshold=0.05):
        df_copy = df.copy()
        encoding_rules = {}
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            value_counts = df_copy[col].value_counts(normalize=True)
            frequent_categories = value_counts[value_counts >= threshold].index.tolist()
            encoding_rules[col] = frequent_categories
        
        return encoding_rules

    def apply_encoding_rules(self, df, encoding_rules):
        df_encoded = df.copy()
        
        for col, rules in encoding_rules.items():
            # Apply 'Other' category to infrequent values
            df_encoded[col] = df[col].apply(lambda x: x if x in rules else 'Other')

            # Create a new DataFrame with the correct columns for one-hot encoding
            encoded_features = pd.get_dummies(df_encoded[col], prefix=col)
            # Ensure all columns that were created during training are present, initialized to False
            all_categories = {f"{col}_{category}": False for category in rules + ['Other']}
            for c in encoded_features.columns:
                all_categories[c] = encoded_features[c].astype(bool)
            encoded_df = pd.DataFrame(all_categories, index=df_encoded.index)

            # Drop the original column and append the new encoded columns
            df_encoded.drop(columns=[col], inplace=True)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

        # Correcting the data types
        for col in df_encoded.columns:
            if 'category' in col:
                df_encoded[col] = df_encoded[col].astype(bool)

        return df_encoded






