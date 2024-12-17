import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class Dataset:
    def __init__(self, data, target, feature_names):
        self.data = data
        self.target = target
        self.feature_names = feature_names

def load_diamonds(filepath="../datasets/diamonds.csv"):
    """
    Load the diamonds dataset and encode categorical features appropriately.
    - Uses Ordinal Encoding for ordered features.
    - Uses One-Hot Encoding for nominal features.
    """
    df = pd.read_csv(filepath)

    # Extract target and features
    target = df['price'].values
    features = df.drop(columns=['price'])

    # Define ordered features and their respective order
    ordinal_features = {
        'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
        'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
        'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    }

    # Ordinal Encoding for ordered features (to preserve the order)
    for col, order in ordinal_features.items():
        if col in features.columns:
            encoder = OrdinalEncoder(categories=[order])
            features[col] = encoder.fit_transform(features[[col]])

    # One-Hot Encoding for nominal features (in order to not "rank" the values)
    nominal_features = features.select_dtypes(include='object').columns.difference(ordinal_features.keys())
    features = pd.get_dummies(features, columns=nominal_features)
    
    return Dataset(features.values, target, list(features.columns))
