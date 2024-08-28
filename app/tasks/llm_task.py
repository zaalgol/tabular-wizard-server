import re
import pandas as pd
from openai import OpenAI
import app.app as app

CODE6 ="import pandas as pd\nimport numpy as np\n\ndef feature_engineering(df):\n    # Convert to datetime format\n    df['TxnDatetime'] = pd.to_datetime(df['TxnDate'] + ' ' + df['TxnTime'], format='%d %b %Y %H:%M:%S')\n    \n    # Drop the original date and time columns since we have combined them\n    df.drop(columns=['TxnDate', 'TxnTime'], inplace=True)\n    \n    # Convert relevant columns to numeric types, handle non-numeric values by converting them to NaN\n    df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce')\n    \n    # Fill NaN values with 0\n    df.fillna(0, inplace=True)\n    \n    # Generate time-based features\n    df['year'] = df['TxnDatetime'].dt.year\n    df['month'] = df['TxnDatetime'].dt.month\n    df['quarter'] = df['TxnDatetime'].dt.quarter\n    df['hour'] = df['TxnDatetime'].dt.hour\n    df['day_of_year'] = df['TxnDatetime'].dt.dayofyear\n    df['week_of_year'] = df['TxnDatetime'].dt.isocalendar().week\n\n    # Create lag features for the specified columns (up to 12 lags)\n    for lag in range(1, 13):\n        df[f'Consumption_lag_{lag}'] = df['Consumption'].shift(lag)\n    \n    # Create rolling window features (mean and sum) for the specified columns with windows of 3, 6, and 12 periods\n    for window in [3, 6, 12]:\n        df[f'Consumption_roll_mean_{window}'] = df['Consumption'].rolling(window).mean()\n        df[f'Consumption_roll_sum_{window}'] = df['Consumption'].rolling(window).sum()\n    \n    # Fill NaN values with 0 again post feature creation to ensure consistency\n    df.fillna(0, inplace=True)\n    \n    return df\n\n"
CODE = "import pandas as pd\n\ndef feature_engineering(df):\n    # Convert 'Period' to datetime\n    df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')\n    \n    # Set 'Period' as the index\n    df.set_index('Period', inplace=True)\n    \n    # Generate time-based features\n    df['Year'] = df.index.year\n    df['Month'] = df.index.month\n    df['Quarter'] = df.index.quarter\n    df['Day_of_Year'] = df.index.dayofyear\n    df['Week_of_Year'] = df.index.isocalendar().week\n    \n    # Handle missing values by filling with appropriate placeholders\n    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)\n    df['Sales_quantity'] = pd.to_numeric(df['Sales_quantity'], errors='coerce').fillna(0)\n    df['Average_cost'] = pd.to_numeric(df['Average_cost'], errors='coerce').fillna(0)\n    df['The_average_annual_payroll_of_the_region'] = pd.to_numeric(df['The_average_annual_payroll_of_the_region'], errors='coerce').fillna(0)\n    \n    # Create lag features\n    for lag in range(1, 4):\n        df[f'Revenue_lag_{lag}'] = df['Revenue'].shift(lag)\n        df[f'Sales_quantity_lag_{lag}'] = df['Sales_quantity'].shift(lag)\n        df[f'Average_cost_lag_{lag}'] = df['Average_cost'].shift(lag)\n        df[f'Payroll_lag_{lag}'] = df['The_average_annual_payroll_of_the_region'].shift(lag)\n    \n    # Create rolling window features\n    for win in [3, 6, 12]:\n        df[f'Revenue_rolling_mean_{win}'] = df['Revenue'].rolling(window=win).mean()\n        df[f'Sales_quantity_rolling_mean_{win}'] = df['Sales_quantity'].rolling(window=win).mean()\n        df[f'Average_cost_rolling_mean_{win}'] = df['Average_cost'].rolling(window=win).mean()\n        df[f'Payroll_rolling_mean_{win}'] = df['The_average_annual_payroll_of_the_region'].rolling(window=win).mean()\n        \n        df[f'Revenue_rolling_sum_{win}'] = df['Revenue'].rolling(window=win).sum()\n        df[f'Sales_quantity_rolling_sum_{win}'] = df['Sales_quantity'].rolling(window=win).sum()\n        df[f'Average_cost_rolling_sum_{win}'] = df['Average_cost'].rolling(window=win).sum()\n        df[f'Payroll_rolling_sum_{win}'] = df['The_average_annual_payroll_of_the_region'].rolling(window=win).sum()\n    \n    # Fill any remaining NaN values\n    df.fillna(0, inplace=True)\n    \n    # Reset the index before returning\n    df.reset_index(inplace=True)\n\n    return df"
CODE2 = "import pandas as pd\n\ndef feature_engineering(df):\n    # Convert 'Period' to datetime type\n    df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')\n    \n    # Set 'Period' as the DataFrame index\n    df.set_index('Period', inplace=True)\n    \n    # Generate time-based features\n    df['Year'] = df.index.year\n    df['Month'] = df.index.month\n    df['Quarter'] = df.index.quarter\n    \n    # Create lag features for 'Revenue', 'Sales_quantity', 'Average_cost', and 'The_average_annual_payroll_of_the_region'\n    for lag in range(1, 13):  # Lag from 1 to 12 months\n        df[f'Revenue_lag_{lag}'] = df['Revenue'].shift(lag)\n        df[f'Sales_quantity_lag_{lag}'] = df['Sales_quantity'].shift(lag)\n        df[f'Average_cost_lag_{lag}'] = df['Average_cost'].shift(lag)\n        df[f'Payroll_lag_{lag}'] = df['The_average_annual_payroll_of_the_region'].shift(lag)\n    \n    # Create rolling window features for 'Revenue', 'Sales_quantity', 'Average_cost', and 'The_average_annual_payroll_of_the_region'\n    for window in [3, 6, 12]:  # Rolling windows of 3, 6, and 12 months\n        df[f'Revenue_rolling_mean_{window}'] = df['Revenue'].rolling(window=window).mean()\n        df[f'Sales_quantity_rolling_mean_{window}'] = df['Sales_quantity'].rolling(window=window).mean()\n        df[f'Average_cost_rolling_mean_{window}'] = df['Average_cost'].rolling(window=window).mean()\n        df[f'Payroll_rolling_mean_{window}'] = df['The_average_annual_payroll_of_the_region'].rolling(window=window).mean()\n    \n    # Handle missing values by filling forward, then filling any remaining NaN values with zeros\n    df.fillna(method='ffill', inplace=True)\n    df.fillna(0, inplace=True)\n    \n    return df\n\n"
CODE3 = 'import pandas as pd\n\ndef feature_engineering(df):\n    """\n    Perform feature engineering on a DataFrame to predict the \'Revenue\' column.\n    \n    Parameters:\n    df (pd.DataFrame): Input DataFrame containing the raw data.\n    \n    Returns:\n    df (pd.DataFrame): DataFrame with engineered features.\n    """\n    # Step 1: Convert the \'Period\' column to datetime format\n    df[\'Period\'] = pd.to_datetime(df[\'Period\'], format=\'%d.%m.%Y\')\n    \n    # Step 2: Set the \'Period\' column as the DataFrame index\n    df.set_index(\'Period\', inplace=True)\n\n    # Step 3: Convert all relevant columns to numeric types\n    relevant_columns = [\'Revenue\', \'Sales_quantity\', \'Average_cost\', \'The_average_annual_payroll_of_the_region\']\n    df[relevant_columns] = df[relevant_columns].apply(pd.to_numeric, errors=\'coerce\')\n\n    # Step 4: Fill any NaN values with 0\n    df.fillna(0, inplace=True)\n\n    # Step 5: Generate time-based features\n    df[\'Year\'] = df.index.year\n    df[\'Month\'] = df.index.month\n    df[\'Quarter\'] = df.index.quarter\n    df[\'Day_of_Year\'] = df.index.dayofyear\n    df[\'Week_of_Year\'] = df.index.isocalendar().week\n\n    # Step 6: Create lag features (up to 12 lags)\n    lag_features = [\'Revenue\', \'Sales_quantity\', \'Average_cost\']\n    for lag in range(1, 13):\n        for feature in lag_features:\n            df[f\'{feature}_lag_{lag}\'] = df[feature].shift(lag)\n\n    # Step 7: Create rolling window features (mean and sum) for the specified columns\n    rolling_windows = [3, 6, 12]\n    for window in rolling_windows:\n        for feature in lag_features:\n            df[f\'{feature}_rolling_mean_{window}\'] = df[feature].rolling(window=window).mean()\n            df[f\'{feature}_rolling_sum_{window}\'] = df[feature].rolling(window=window).sum()\n\n    # Fill NaNs produced by rolling operations by 0\n    df.fillna(0, inplace=True)\n\n    # Ensure Revenue_Bucket is not NaN after all operations (It shouldn\'t be in general cases)\n    if \'Revenue_Bucket\' in df.columns:\n        df[\'Revenue_Bucket\'].fillna(\'Unknown\', inplace=True)\n\n    return df'
CODE4 = "import pandas as pd\n\ndef feature_engineering(df):\n    # Step 1: Convert 'Period' column to datetime format\n    df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')\n    \n    # Step 2: Set the 'Period' column as the DataFrame index\n    df.set_index('Period', inplace=True)\n    \n    # Step 3: Convert relevant columns to numeric types\n    columns_to_convert = ['Revenue', 'Sales_quantity', 'Average_cost', \n                          'The_average_annual_payroll_of_the_region']\n    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')\n    \n    # Fill NaN values in relevant columns with 0\n    df.fillna(0, inplace=True)\n    \n    # Step 5: Generate time-based features\n    df['Year'] = df.index.year\n    df['Month'] = df.index.month\n    df['Quarter'] = df.index.quarter\n    df['Day_of_Year'] = df.index.dayofyear\n    df['Week_of_Year'] = df.index.isocalendar().week\n    \n    # Step 6: Create lag features (up to 12 lags)\n    lags = 12\n    for lag in range(1, lags + 1):\n        for column in columns_to_convert:\n            df[f'{column}_lag_{lag}'] = df[column].shift(lag)\n            \n    # Step 7: Create rolling window features (mean and sum)\n    window_sizes = [3, 6, 12]\n    for window in window_sizes:\n        for column in columns_to_convert:\n            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()\n            df[f'{column}_rolling_sum_{window}'] = df[column].rolling(window=window).sum()\n    \n    # Return the processed dataset\n    return df\n\n"
CODE5 = "import pandas as pd\n\ndef feature_engineering(df):\n    # Step 1: Convert 'Period' column to datetime format\n    df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')\n    \n    # Step 2: Set 'Period' column as the DataFrame index\n    df.set_index('Period', inplace=True)\n    \n    # Step 3: Convert all relevant columns to numeric types\n    for column in df.columns:\n        df[column] = pd.to_numeric(df[column], errors='coerce')\n        \n    # Step 4: Fill any NaN values with 0\n    df.fillna(0, inplace=True)\n    \n    # Step 5: Generate time-based features\n    df['year'] = df.index.year\n    df['month'] = df.index.month\n    df['quarter'] = df.index.quarter\n    df['day_of_year'] = df.index.dayofyear\n    df['week_of_year'] = df.index.isocalendar().week\n    \n    # List of columns for which we want to create lag and rolling features\n    columns_to_lag_and_roll = ['Revenue', 'Sales_quantity', 'Average_cost']\n    \n    # Step 6: Create lag features (up to 12 lags)\n    for column in columns_to_lag_and_roll:\n        for lag in range(1, 13):\n            df[f'{column}_lag_{lag}'] = df[column].shift(lag)\n    \n    # Step 7: Create rolling window features (mean and sum) with windows of 3, 6, and 12 periods\n    for column in columns_to_lag_and_roll:\n        for window in [3, 6, 12]:\n            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()\n            df[f'{column}_rolling_sum_{window}'] = df[column].rolling(window=window).sum()\n    \n    return df\n\n"

class LlmTask:
    def __init__(self) -> None:
        self.api_key = app.Config.OPENAI_API_KEY
        self.model = app.Config.MODEL
        self.max_tokens = app.Config.MAX_TOKENS
        self.llm_max_tries = app.Config.LLM_MAX_TRIES
        
        
        

    def use_llm_toproccess_timeseries_dataset(self, raw_dataset, target_column):
        # tt = self.feature_engineering(raw_dataset)
        try_no = 1
        try:
            head_rows = raw_dataset.head(35).to_csv(index=False)
            tail_rows = raw_dataset.tail(35).to_csv(index=False)
            
            def use_llm_toproccess_timeseries_dataset_execution():
                # Get the feature engineering code from LLM model
                code = CODE4
                code = self._get_feature_engineering_code(head_rows, tail_rows, target_column)
                
                
                processed_dataset = self.processed_dataset(raw_dataset, code)
            
                # # Return the processed dataset
                # processed_dataset = local_vars.get('processed_dataset', raw_dataset)
                print(f"processed_dataset: {processed_dataset}")
                return processed_dataset, code
            return use_llm_toproccess_timeseries_dataset_execution()
        except Exception as e:
            print(f"{e}")
            if try_no < self.llm_max_tries:
                try_no += 1
                return use_llm_toproccess_timeseries_dataset_execution()
                
                
        
        
    def processed_dataset(self, raw_dataset, code):
        
        local_vars = {}
    
        # Execute the code within the context of local_vars
        exec(code, {"pd": pd}, local_vars)
        
        # Retrieve the feature_engineering function from local_vars
        feature_engineering = local_vars['feature_engineering']
        processed_dataset = feature_engineering(raw_dataset)
        return processed_dataset
    
    def _get_feature_engineering_code(self, head_rows, tail_rows, target_column):
        # Combine the head and tail rows into a single prompt
        combined_data = f"head of 35 rows:\n{head_rows}\ntail of 35 rows:\n{tail_rows}"
        prompt = (
            f"Given the following raw time series dataset:\n{combined_data}\n"
            f"Please write a Python function named feature_engineering with a parameter named df, that performs feature engineering to predict the '{target_column}' column. "
            "The function should include the following steps:\n"
            "1. Convert the datetime, date, and time columns to datetime format.\n"
            "2. Convert all relevant columns to numeric types, handling non-numeric values by converting them to NaN.\n"
            "3. Fill any NaN values in the target column with the previous value (forward fill) if the target column is missing, and for the last row, fill it with 0 to ensure the function works during inference.\n"
            "4. Generate time-based features such as year, month, quarter, hour, day of year, and week of year.\n"
            "5. Create lag features for the specified columns (up to 12 lags).\n"
            "6. Create rolling window features (mean and sum) for the specified columns with windows of 3, 6, and 12 periods.\n"
            "Ensure that the generated code is safe and does not perform any harmful or malicious actions. "
            "The function should not read from any files and should return the processed dataset directly as a variable named 'df'. "
            "Do not include any data scaling or transformations. Handle missing values appropriately to ensure columns are treated as numeric types, "
            "and fill missing values to keep the number of rows consistent."
        )
        client = OpenAI(
            # This is the default and can be omitted
            api_key=self.api_key
        )
        try:
            # Use OpenAI API to get the feature engineering code
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
            )
            
            # Extract the code from the response
            code = re.search(r'```python\n(.*?)\n```', response.choices[0].message.content, re.DOTALL).group(1)
            code = code.split("# Example usage")[0]
            print(f"code: {code}")
            return code
        except Exception as e:
            print(f"{e}")