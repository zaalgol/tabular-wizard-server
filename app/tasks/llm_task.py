import re
import pandas as pd
from openai import OpenAI
import app.app as app

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
            
            # Get the feature engineering code from GPT-4
            code = self._get_feature_engineering_code(head_rows, tail_rows, target_column)
            
            processed_dataset = self.processed_dataset(raw_dataset, code)
            # Execute the returned code
            # local_vars = {'pd': pd, 'raw_dataset': raw_dataset.copy()}
            # exec(code, {}, local_vars)
            
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
        prompt = (
            f"Given the following raw time series dataset:\n{combined_data}\n"
            f"Please write a Python function named feature_engineering with a parameter named df, that performs feature engineering to predict the '{target_column}' column. "
            "The function should include parsing dates, generating time-based features, creating lag features, and rolling window features. "
            "Ensure that the generated code is safe and does not perform any harmful or malicious actions. "
            "The function should not read from any files and should return the processed dataset directly as a variable named 'df'. "
            "Do not include any data scaling or transformations."
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