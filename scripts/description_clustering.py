import re
import os
import numpy as np
import pandas as pd
from pandas import json_normalize

import torch
from transformers import BertTokenizer, BertModel

import umap
from hdbscan import HDBSCAN

class DescriptionCluster:
    def __init__(self, input_path, output_path, min_cluster_size=10):
        self.input_path = input_path
        self.output_path = output_path
        self.min_cluster_size = min_cluster_size

    @staticmethod
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # Emoticons
                                   u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # Alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def load_data(self):
        try:
            df = pd.read_json(self.input_path)
            return df
        except Exception as e:
            print(f"Error reading the JSON file: {str(e)}")
            exit(1)

    def preprocess_data(self, df):
        try:
            df['translated_description'] = df['translated_description'].apply(self.preprocess_text)
            return df
        except KeyError:
            print("The 'translated_description' column is missing in the JSON file.")
            exit(1)

    @staticmethod
    def get_bert_embeddings(text, tokenizer, model):
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].numpy()  
        except Exception as e:
            print(f"Error getting BERT embeddings: {str(e)}")
            return np.zeros((1, 768))  

    def generate_embeddings(self, df):
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(f"Error loading BERT model or tokenizer: {str(e)}")
            exit(1)
        try:
            embeddings = np.vstack([self.get_bert_embeddings(text, tokenizer, model) for text in df['translated_description']])
            return embeddings
        except Exception as e:
            print(f"Error generating BERT embeddings: {str(e)}")
            exit(1)

    def reduce_dimensionality(self, embeddings):
        try:
            reducer = umap.UMAP(n_components=128, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            return reduced_embeddings
        except Exception as e:
            print(f"Error during UMAP dimensionality reduction: {str(e)}")
            exit(1)

    def cluster_data(self, reduced_embeddings):
        try:
            clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size)
            labels = clusterer.fit_predict(reduced_embeddings)
            return labels
        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            exit(1)

    def save_data(self, df):
        try:
            df.to_json(self.output_path, orient='records', indent=4)
            print(f"Output saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving the JSON file: {str(e)}")

    def process(self):
        df = self.load_data()
        df = self.preprocess_data(df)
        embeddings = self.generate_embeddings(df)
        reduced_embeddings = self.reduce_dimensionality(embeddings)
        labels = self.cluster_data(reduced_embeddings)
        df['description_cluster'] = labels
        self.save_data(df)


if __name__ == "__main__":
    input_path = "D:/instagramproject/json files/Loksabha_t.json"
    output_path = "D:/instagramproject/json files/Loksabha_t_descCluster.json"
    clusterer = DescriptionCluster(input_path, output_path, min_cluster_size=10)
    clusterer.process()
