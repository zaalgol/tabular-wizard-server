import traceback
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

class NlpEmbeddingsPreprocessing:
    def __init__(self, model_name='bert-base-uncased', max_length=512, device=None):
        self.model_name = model_name
        self.max_length = max_length
        # Automatically select device if not specified
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
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
            'columns': embedding_columns,
            'device': self.device
        }

        # Calculate embeddings for the columns and store the dimension size
        sample_text = df[embedding_columns[0]].iloc[0] if len(df) > 0 else ''
        embedding_sample = self._get_bert_embedding(sample_text)
        embedding_rules['embedding_dimensions'] = len(embedding_sample)

        return embedding_rules

<<<<<<< HEAD
   
    def apply_embedding_rules(self, df: pd.DataFrame, embedding_rules: dict) -> pd.DataFrame:
=======
    def apply_embedding_rules(self, df: pd.DataFrame, embedding_rules: dict, batch_size: int = 32) -> pd.DataFrame:
>>>>>>> main
        """
        Apply BERT embeddings based on the previously created embedding rules.
        
        :param df: Input pandas DataFrame containing the text columns.
        :param embedding_rules: Dictionary containing the embedding rules.
        :param batch_size: Number of texts to process at once (to optimize GPU memory usage)
        :return: DataFrame with BERT embeddings added.
        """
        print("Starting applying embeddings")
        df_copy = df.copy()
        embedding_columns = embedding_rules['columns']
        model_name = embedding_rules['model_name']
        max_length = embedding_rules['max_length']

        try:
            # Re-initialize the model and tokenizer if necessary
            if (self.model_name != model_name or 
                self.max_length != max_length or 
                self.device != embedding_rules.get('device')):
                
                self.model_name = model_name
                self.max_length = max_length
                self.device = embedding_rules.get('device', self.device)
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()

            for col in embedding_columns:
                # Convert categorical data to strings
                df_copy[col] = df_copy[col].astype(str)
                
                # Process in batches
                embeddings_list = []
                for i in range(0, len(df_copy), batch_size):
                    batch_texts = df_copy[col].iloc[i:i+batch_size].tolist()
                    batch_embeddings = self._get_bert_embeddings_batch(batch_texts)
                    embeddings_list.extend(batch_embeddings)

                # Create DataFrame with embeddings
                embeddings_df = pd.DataFrame(embeddings_list, index=df_copy.index)
                embeddings_df.columns = [f'{col}_bert_{i}' for i in range(embeddings_df.shape[1])]
                df_copy = pd.concat([df_copy, embeddings_df], axis=1)
                df_copy.drop(columns=[col], inplace=True)

        except Exception as e:
            print(f"Error during embedding generation: {str(e)}")
            print(traceback.format_exc())
            raise
            
        print("Ending applying embeddings")
        return df_copy

    def _get_bert_embeddings_batch(self, texts: list) -> list:
        """
        Generate BERT embeddings for a batch of texts.
        
        :param texts: List of input texts.
        :return: List of numpy arrays containing the averaged BERT embeddings.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Move input tensors to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            # Average the embeddings of all tokens to get fixed-size vectors
            embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.tolist()

    def _get_bert_embedding(self, text: str):
        """
        Generate BERT embeddings for a single piece of text.
        
        :param text: Input text.
        :return: Numpy array containing the averaged BERT embeddings.
        """
        return self._get_bert_embeddings_batch([text])[0]

# Example Usage:
# handler = NlpEmbeddingsPreprocessing(device='cuda')  # Explicitly specify GPU
# df = pd.DataFrame({'comments': ['This is great!', 'Needs improvement.', 'Absolutely fantastic!']})
# embedding_rules = handler.create_embedding_rules(df, ['comments'])
# df_with_embeddings = handler.apply_embedding_rules(df, embedding_rules, batch_size=32)
# print(df_with_embeddings.head())