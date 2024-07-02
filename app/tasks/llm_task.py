import re
import pandas as pd
from openai import OpenAI
import app.app as app

CODE = "import pandas as pd\n\ndef feature_engineering(df):\n    # Convert 'Period' to datetime\n    df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')\n    \n    # Set 'Period' as the index\n    df.set_index('Period', inplace=True)\n    \n    # Generate time-based features\n    df['Year'] = df.index.year\n    df['Month'] = df.index.month\n    df['Quarter'] = df.index.quarter\n    df['Day_of_Year'] = df.index.dayofyear\n    df['Week_of_Year'] = df.index.isocalendar().week\n    \n    # Handle missing values by filling with appropriate placeholders\n    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)\n    df['Sales_quantity'] = pd.to_numeric(df['Sales_quantity'], errors='coerce').fillna(0)\n    df['Average_cost'] = pd.to_numeric(df['Average_cost'], errors='coerce').fillna(0)\n    df['The_average_annual_payroll_of_the_region'] = pd.to_numeric(df['The_average_annual_payroll_of_the_region'], errors='coerce').fillna(0)\n    \n    # Create lag features\n    for lag in range(1, 4):\n        df[f'Revenue_lag_{lag}'] = df['Revenue'].shift(lag)\n        df[f'Sales_quantity_lag_{lag}'] = df['Sales_quantity'].shift(lag)\n        df[f'Average_cost_lag_{lag}'] = df['Average_cost'].shift(lag)\n        df[f'Payroll_lag_{lag}'] = df['The_average_annual_payroll_of_the_region'].shift(lag)\n    \n    # Create rolling window features\n    for win in [3, 6, 12]:\n        df[f'Revenue_rolling_mean_{win}'] = df['Revenue'].rolling(window=win).mean()\n        df[f'Sales_quantity_rolling_mean_{win}'] = df['Sales_quantity'].rolling(window=win).mean()\n        df[f'Average_cost_rolling_mean_{win}'] = df['Average_cost'].rolling(window=win).mean()\n        df[f'Payroll_rolling_mean_{win}'] = df['The_average_annual_payroll_of_the_region'].rolling(window=win).mean()\n        \n        df[f'Revenue_rolling_sum_{win}'] = df['Revenue'].rolling(window=win).sum()\n        df[f'Sales_quantity_rolling_sum_{win}'] = df['Sales_quantity'].rolling(window=win).sum()\n        df[f'Average_cost_rolling_sum_{win}'] = df['Average_cost'].rolling(window=win).sum()\n        df[f'Payroll_rolling_sum_{win}'] = df['The_average_annual_payroll_of_the_region'].rolling(window=win).sum()\n    \n    # Fill any remaining NaN values\n    df.fillna(0, inplace=True)\n    \n    # Reset the index before returning\n    df.reset_index(inplace=True)\n\n    return df"
CODE2 = "import pandas as pd\n\ndef feature_engineering(df):\n    # Convert 'Period' to datetime type\n    df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')\n    \n    # Set 'Period' as the DataFrame index\n    df.set_index('Period', inplace=True)\n    \n    # Generate time-based features\n    df['Year'] = df.index.year\n    df['Month'] = df.index.month\n    df['Quarter'] = df.index.quarter\n    \n    # Create lag features for 'Revenue', 'Sales_quantity', 'Average_cost', and 'The_average_annual_payroll_of_the_region'\n    for lag in range(1, 13):  # Lag from 1 to 12 months\n        df[f'Revenue_lag_{lag}'] = df['Revenue'].shift(lag)\n        df[f'Sales_quantity_lag_{lag}'] = df['Sales_quantity'].shift(lag)\n        df[f'Average_cost_lag_{lag}'] = df['Average_cost'].shift(lag)\n        df[f'Payroll_lag_{lag}'] = df['The_average_annual_payroll_of_the_region'].shift(lag)\n    \n    # Create rolling window features for 'Revenue', 'Sales_quantity', 'Average_cost', and 'The_average_annual_payroll_of_the_region'\n    for window in [3, 6, 12]:  # Rolling windows of 3, 6, and 12 months\n        df[f'Revenue_rolling_mean_{window}'] = df['Revenue'].rolling(window=window).mean()\n        df[f'Sales_quantity_rolling_mean_{window}'] = df['Sales_quantity'].rolling(window=window).mean()\n        df[f'Average_cost_rolling_mean_{window}'] = df['Average_cost'].rolling(window=window).mean()\n        df[f'Payroll_rolling_mean_{window}'] = df['The_average_annual_payroll_of_the_region'].rolling(window=window).mean()\n    \n    # Handle missing values by filling forward, then filling any remaining NaN values with zeros\n    df.fillna(method='ffill', inplace=True)\n    df.fillna(0, inplace=True)\n    \n    return df\n\n"
class LlmTask:
    def __init__(self) -> None:
        self.api_key = app.Config.OPENAI_API_KEY
        self.model = app.Config.MODEL
        self.max_tokens = app.Config.MAX_TOKENS
        
        
        

    def use_llm_toproccess_timeseries_dataset(self, raw_dataset, target_column):
        # tt = self.feature_engineering(raw_dataset)

        try:
            head_rows = raw_dataset.head(35).to_csv(index=False)
            tail_rows = raw_dataset.tail(35).to_csv(index=False)
            
            # Get the feature engineering code from LLM model
            code = self._get_feature_engineering_code(head_rows, tail_rows, target_column)
            # code = CODE
            
            processed_dataset = self.processed_dataset(raw_dataset, code)
            
            # # Return the processed dataset
            # processed_dataset = local_vars.get('processed_dataset', raw_dataset)
            return processed_dataset, code
        except Exception as e:
            print(f"{e}")
        
        
    def processed_dataset(self, raw_dataset, code):
        # Execute the returned code
        # local_vars = {'pd': pd, 'df': raw_dataset.copy()}
        # exec(code, {}, local_vars)
        
        # Return the processed dataset
        # processed_dataset = local_vars.get('df', raw_dataset)
        
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
        # prompt = (
        #     f"Given the following raw time series dataset:\n{combined_data}\n"
        #     f"Please write a Python function named feature_engineering with a parameter named df, that performs feature engineering to predict the '{target_column}' column. "
        #     "The function should include parsing dates, generating time-based features, creating lag features, and rolling window features. "
        #     "Ensure that the generated code is safe and does not perform any harmful or malicious actions. "
        #     "The function should not read from any files and should return the processed dataset directly as a variable named 'df'. "
        #     "Do not include any data scaling or transformations. Handle missing values appropriately to ensure columns are treated as numeric types, "
        #     "and fill missing values to keep the number of rows consistent."
        # )
        prompt = (
            f"Given the following raw time series dataset:\n{combined_data}\n"
            f"Please write a Python function named feature_engineering with a parameter named df, that performs feature engineering to predict the '{target_column}' column. "
            "The function should include the following steps:\n"
            "1. Convert the 'Period' column to datetime format.\n"
            "2. Set the 'Period' column as the DataFrame index.\n"
            "3. Convert all relevant columns to numeric types, handling non-numeric values by converting them to NaN.\n"
            "4. Fill any NaN values with 0 before creating lag and rolling window features to ensure consistency.\n"
            "5. Generate time-based features such as year, month, quarter, day of year, and week of year.\n"
            "6. Create lag features for the specified columns (up to 12 lags).\n"
            "7. Create rolling window features (mean and sum) for the specified columns with windows of 3, 6, and 12 periods.\n"
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
            # code = re.sub(r'\braw_data\b', 'df', code)  # Replace all occurrences of raw_data with df
    
            # code = code.replace("df = pd.read_csv('<your file path here>')", "")
            # code = code.replace("# Read data from your file", "")
            # code = code.replace("# Execute function", "")
            # code = code.replace("processed_df = process_timeseries(df)", "processed_df = process_timeseries(raw_dataset)")
            return code
        except Exception as e:
            print(f"{e}")