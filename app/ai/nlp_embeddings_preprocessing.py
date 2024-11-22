import traceback
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

class NlpEmbeddingsPreprocessing:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval()  # Set the model to evaluation mode

    def create_embedding_rules(self, df: pd.DataFrame, embedding_columns: list) -> dict:
        """
        Create embedding rules for the specified columns using BERT embeddings.
        
        :param df: Input pandas DataFrame containing the text columns.
        :param embedding_columns: List of columns to apply BERT embeddings.
        :return: Dictionary containing the embedding rules and model configurations.
        """
        embedding_rules = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'embedding_dimensions': None,  # Will be determined during embedding creation
            'columns': embedding_columns
        }

        # Calculate embeddings for the columns and store the dimension size
        sample_text = df[embedding_columns[0]].iloc[0] if len(df) > 0 else ''
        embedding_sample = self._get_bert_embedding(sample_text)
        embedding_rules['embedding_dimensions'] = len(embedding_sample)

        return embedding_rules

    # def apply_embedding_rules(self, df: pd.DataFrame, embedding_rules: dict) -> pd.DataFrame:
    #     """
    #     Apply BERT embeddings based on the previously created embedding rules.
        
    #     :param df: Input pandas DataFrame containing the text columns.
    #     :param embedding_rules: Dictionary containing the embedding rules.
    #     :return: DataFrame with BERT embeddings added.
    #     """
    #     df_copy = df.copy()
    #     embedding_columns = embedding_rules['columns']
    #     model_name = embedding_rules['model_name']
    #     max_length = embedding_rules['max_length']

    #     try:
    #         # Re-initialize the model and tokenizer if necessary
    #         if self.model_name != model_name or self.max_length != max_length:
    #             self.model_name = model_name
    #             self.max_length = max_length
    #             self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
    #             self.model = BertModel.from_pretrained(self.model_name)
    #             self.model.eval()
    #         for col in embedding_columns:
    #             # Apply BERT embeddings and append new columns for each embedding feature
    #             embeddings = df_copy[col].apply(lambda x: self._get_bert_embedding(str(x)))
    #             embeddings_df = pd.DataFrame(embeddings.tolist(), index=df_copy.index)
    #             embeddings_df.columns = [f'{col}_bert_{i}' for i in range(embeddings_df.shape[1])]
    #             df_copy = pd.concat([df_copy, embeddings_df], axis=1)
    #             df_copy.drop(columns=[col], inplace=True)  # Optionally drop the original text column

    #     except Exception as e:
    #         print(e)
    #         print(traceback.format_exc())

    #     return df_copy
    def apply_embedding_rules(self, df: pd.DataFrame, embedding_rules: dict) -> pd.DataFrame:
        """
        Apply BERT embeddings based on the previously created embedding rules.
        
        :param df: Input pandas DataFrame containing the text columns.
        :param embedding_rules: Dictionary containing the embedding rules.
        :return: DataFrame with BERT embeddings added.
        """
        print("starting applying embeddings")
        df_copy = df.copy()
        embedding_columns = embedding_rules['columns']
        model_name = embedding_rules['model_name']
        max_length = embedding_rules['max_length']

        try:
            # Re-initialize the model and tokenizer if necessary
            if self.model_name != model_name or self.max_length != max_length:
                self.model_name = model_name
                self.max_length = max_length
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
                self.model.eval()

            for col in embedding_columns:
                # Convert categorical data to strings
                df_copy[col] = df_copy[col].astype(str)

                # Apply BERT embeddings and append new columns for each embedding feature
                embeddings = df_copy[col].apply(lambda x: self._get_bert_embedding(str(x)))
                embeddings_df = pd.DataFrame(embeddings.tolist(), index=df_copy.index)
                embeddings_df.columns = [f'{col}_bert_{i}' for i in range(embeddings_df.shape[1])]
                df_copy = pd.concat([df_copy, embeddings_df], axis=1)
                df_copy.drop(columns=[col], inplace=True)  # Optionally drop the original text column

        except Exception as e:
            print(traceback.format_exc())
        print("ending applying embeddings")
        return df_copy


    def _get_bert_embedding(self, text: str):
        """
        Generate BERT embeddings for a single piece of text.
        
        :param text: Input text.
        :return: Numpy array containing the averaged BERT embeddings.
        """
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            # Average the embeddings of all tokens to get a fixed-size vector
            embeddings = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embeddings

# Example Usage:
# handler = BertEmbeddingHandler()
# df = pd.DataFrame({'comments': ['This is great!', 'Needs improvement.', 'Absolutely fantastic!']})
# embedding_rules = handler.create_embedding_rules(df, ['comments'])
# df_with_embeddings = handler.apply_embedding_rules(df, embedding_rules)
# print(df_with_embeddings.head())
