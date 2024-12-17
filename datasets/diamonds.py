import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, data, target, feature_names):
        self.data = data
        self.target = target
        self.feature_names = feature_names

def load_diamonds(filepath="../datasets/diamonds.csv"):
    """
    Load the diamonds dataset in a format similar to sklearn's dataset loaders.
    """
    df = pd.read_csv(filepath)

    # Extract features and target
    target = df['price'].values
    data = df.drop(columns=['price']).values
    
    return Dataset(data, target, list(df.columns[:-1]))
