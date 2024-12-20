import re
import pandas as pd
import ast
from openai import OpenAI
import app.app as app

# CODE = "def feature_engineering(df):\n    # Step 1: Convert the 'Period' column to datetime format\n    df['Period'] = pd.to_datetime(df['Period'])\n\n    # Step 2: Convert all relevant columns to numeric types, handling non-numeric values by converting them to NaN\n    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')\n    df['Sales_quantity'] = pd.to_numeric(df['Sales_quantity'], errors='coerce')\n    df['Average_cost'] = pd.to_numeric(df['Average_cost'], errors='coerce')\n    df['The_average_annual_payroll_of_the_region'] = pd.to_numeric(df['The_average_annual_payroll_of_the_region'], errors='coerce')\n\n    # Step 3: Fill NaN values in the 'Revenue' column\n    df['Revenue'].fillna(method='ffill', inplace=True)\n    df['Revenue'].fillna(method='bfill', inplace=True)\n\n    # Step 4: Generate time-based features\n    df['year'] = df['Period'].dt.year\n    df['month'] = df['Period'].dt.month\n    df['quarter'] = df['Period'].dt.quarter\n    df['day_of_year'] = df['Period'].dt.dayofyear\n    df['week_of_year'] = df['Period'].dt.isocalendar().week\n    df['day_of_week'] = df['Period'].dt.dayofweek\n\n    # Step 5: Create lag features for the specified columns (up to 12 lags)\n    lag_columns = ['Revenue', 'Sales_quantity', 'Average_cost', 'The_average_annual_payroll_of_the_region']\n    for column in lag_columns:\n        for lag in range(1, 13):\n            df[f'{column}_lag_{lag}'] = df[column].shift(lag).fillna(0)\n\n    # Step 6: Create rolling window features (mean and sum) for the specified columns\n    window_sizes = [3, 6, 12]\n    for column in lag_columns:\n        for window in window_sizes:\n            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean().fillna(0)\n            df[f'{column}_rolling_sum_{window}'] = df[column].rolling(window=window).sum().fillna(0)\n\n    return df"

CODE = "def feature_engineering(df):\n    # Step 1: Convert the datetime column to datetime format\n    df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y', errors='coerce')\n    \n    # Step 2: Convert relevant columns to numeric types\n    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')\n    df['Sales_quantity'] = pd.to_numeric(df['Sales_quantity'], errors='coerce')\n    df['Average_cost'] = pd.to_numeric(df['Average_cost'], errors='coerce')\n    df['The_average_annual_payroll_of_the_region'] = pd.to_numeric(df['The_average_annual_payroll_of_the_region'], errors='coerce')\n    \n    # Step 3: Fill NaN values in the target column 'Revenue'\n    if df['Revenue'].isnull().any():\n        df['Revenue'].fillna(method='ffill', inplace=True)\n        df['Revenue'].fillna(method='bfill', inplace=True)\n\n    # Step 4: Generate time-based features\n    df['year'] = df['Period'].dt.year\n    df['month'] = df['Period'].dt.month\n    df['quarter'] = df['Period'].dt.quarter\n    df['day_of_year'] = df['Period'].dt.dayofyear\n    df['week_of_year'] = df['Period'].dt.isocalendar().week\n\n    # Step 5: Create lag features for the specified columns\n    columns_to_lag = ['Revenue', 'Sales_quantity', 'Average_cost', 'The_average_annual_payroll_of_the_region']\n    for col in columns_to_lag:\n        for lag in range(1, 13):\n            lag_col_name = f'{col}_lag_{lag}'\n            df[lag_col_name] = df[col].shift(lag).fillna(0.0)\n\n    # Step 6: Create rolling window features (mean and sum) for specified columns\n    window_sizes = [3, 6, 12]\n    for col in columns_to_lag:\n        for window in window_sizes:\n            rolling_mean_col = f'{col}_rolling_mean_{window}'\n            rolling_sum_col = f'{col}_rolling_sum_{window}'\n            df[rolling_mean_col] = df[col].rolling(window=window).mean().fillna(0.0)\n            df[rolling_sum_col] = df[col].rolling(window=window).sum().fillna(0.0)\n    \n    return df"
class LlmTask:
    def __init__(self) -> None:
        self.api_key = app.Config.OPENAI_API_KEY
        self.model = app.Config.MODEL
        self.max_tokens = app.Config.MAX_TOKENS
        self.llm_max_tries = app.Config.LLM_MAX_TRIES
        self.llm_number_of_dataset_lines = app.Config.LLM_NUMBER_OF_DATASET_LINES

    def use_llm_to_proccess_timeseries_dataset(self, raw_dataset, target_column):
        # return raw_dataset, CODE
        try_no = 1
        try:
            head_rows = raw_dataset.head(int(self.llm_number_of_dataset_lines)).to_csv(index=False)
            tail_rows = raw_dataset.tail(int(self.llm_number_of_dataset_lines)).to_csv(index=False)

            def use_llm_toproccess_timeseries_dataset_execution():
                # Get the feature engineering code from LLM model
                code = self._get_feature_engineering_code(
                    head_rows, tail_rows, target_column
                )
                print("*" * 100 + "  " + str(code) + "  " + "*" * 100)
                processed_dataset = self.processed_dataset(raw_dataset, code)

                # Return the processed dataset
                print(f"processed_dataset: {processed_dataset}")
                return processed_dataset, code

            return use_llm_toproccess_timeseries_dataset_execution()
        except Exception as e:
            print(f"{e}")
            if try_no < self.llm_max_tries:
                try_no += 1
                return use_llm_toproccess_timeseries_dataset_execution()

    def processed_dataset(self, raw_dataset, code):
        try:
            # Check if the code is safe before executing
            if not self.is_code_safe(code):
                raise Exception("Generated code is not safe to execute.")

            # Prepare the execution environment
            local_vars = {}
            safe_globals = {"pd": pd, "int": int, "float": float, "range": range, "__builtins__": {}}

            # Execute the code within the context of local_vars and safe_globals
            exec(code, safe_globals, local_vars)

            # Retrieve the feature_engineering function from local_vars
            feature_engineering = local_vars["feature_engineering"]
            processed_dataset = feature_engineering(raw_dataset)
            return processed_dataset
        except Exception as e:
            print(e)

    def _get_feature_engineering_code(self, head_rows, tail_rows, target_column):
        # return CODE
        # Sanitize the data to remove any code or malicious content
        head_rows = self.sanitize_data(head_rows)
        tail_rows = self.sanitize_data(tail_rows)

        # Combine the head and tail rows into a single prompt
        combined_data = f"head of {self.llm_number_of_dataset_lines} rows:\n{head_rows}\ntail of {self.llm_number_of_dataset_lines} rows:\n{tail_rows}"
        prompt = (
            f"Given the following raw time series dataset:\n{combined_data}\n"
            f"Please write a Python function named feature_engineering with a parameter named df, that performs feature engineering to predict the '{target_column}' column. "
            "The function should include the following steps:\n"
            "1. Convert the datetime, date, and time columns to datetime format.\n"
            "2. Convert all relevant columns to numeric types, handling non-numeric values by converting them to NaN.\n"
            "3. Fill any NaN values in the target column with the next value (forward fill) if the target column is missing, and for the last row, fill it with the previus vlaue.\n"
            "4. Generate time-based features such as year, month, quarter, hour, day of year, and week of year.\n"
            "5. Create lag features for the specified columns (up to 12 lags). If lag cell value is NaN, convert it to 0.0 \n"
            "6. Create rolling window features (mean and sum) for the specified columns with windows of 3, 6, and 12 periods. If cell value is NaN, convert it to 0.0\n"
            "Ensure that the generated code is safe and does not perform any harmful or malicious actions. "
            "The function should not read from any files and should return the processed dataset directly as a variable named 'df'. "
            "Do not include any data scaling or transformations. Handle missing values appropriately to ensure columns are treated as numeric types, "
            "and fill missing values to keep the number of rows consistent."
            "\nNote: Do not include any import statements in your code; assume all necessary libraries are already imported."
        )
        print(f"prompt:{prompt}")
        client = OpenAI(api_key=self.api_key)
        try:
            # Use OpenAI API to get the feature engineering code
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )

            # Extract the code from the response
            code_match = re.search(
                r"```python\n(.*?)\n```", response.choices[0].message.content, re.DOTALL
            )
            if not code_match:
                raise Exception("Failed to extract code from the response.")

            code = code_match.group(1)
            code = code.split("# Example usage")[0]
            print(f"code: {code}")
            return code
        except Exception as e:
            print(f"{e}")

    def is_code_safe(self, code):
        """
        Check if the provided code is safe to execute.
        """
        try:
            # Parse the code into an Abstract Syntax Tree (AST)
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            return False

        # Check for dangerous nodes in the AST
        if not self._is_ast_safe(tree):
            return False

        return True

    def _is_ast_safe(self, tree):
        """
        Recursively check AST nodes for unsafe constructs.
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Check if import statements are safe
                # if not self._is_import_safe(node):
                    return False
            elif isinstance(node, ast.Call):
                # Check if function calls are safe
                if not self._is_call_safe(node):
                    return False
            elif isinstance(node, ast.Attribute):
                # Check if attribute access is safe
                if not self._is_attribute_safe(node):
                    return False
            elif isinstance(node, ast.Name):
                # Check if variable or function names are safe
                if not self._is_name_safe(node):
                    return False
        return True


    def _is_call_safe(self, node):
        """
        Check if function calls are safe.
        """
        dangerous_functions = {
            'eval', 'exec', 'open', 'compile', 'input',
            'globals', 'locals', 'vars', '__import__'
        }
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in dangerous_functions:
                print(f"Use of dangerous function '{func_name}' detected.")
                return False
        elif isinstance(node.func, ast.Attribute):
            # Check if calling dangerous methods from dangerous modules
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                attr_name = node.func.attr
                dangerous_modules = {'os', 'sys', 'subprocess', 'shutil'}
                if module_name in dangerous_modules:
                    print(f"Use of dangerous module '{module_name}' detected.")
                    return False
        return True

    def _is_attribute_safe(self, node):
        """
        Check if attribute access is safe.
        """
        if isinstance(node.value, ast.Name):
            if node.value.id == '__builtins__':
                print("Access to '__builtins__' is not allowed.")
                return False
        return True

    def _is_name_safe(self, node):
        """
        Check if variable or function names are safe.
        """
        dangerous_names = {
            'eval', 'exec', 'open', 'compile', 'input',
            'globals', 'locals', 'vars', '__import__', '__builtins__'
        }
        if node.id in dangerous_names:
            print(f"Use of dangerous name '{node.id}' detected.")
            return False
        return True

    def sanitize_data(self, data):
        """
        Remove any code-like patterns from the data.
        """
        # Remove any script tags or code injections
        data = re.sub(r'[<>]', '', data)
        data = re.sub(r'(?i)script', '', data)
        data = re.sub(r'(?i)eval\(', '', data)
        return data
