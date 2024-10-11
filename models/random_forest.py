import numpy as np
from models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth, num_thresholds) -> None:
        super().__init__(max_depth, num_thresholds)
    
    # method is called in DecisionTree._best_split
    def _modify_split_data(self, X): 
        # randomly selects the indexes of between 1 and all features of the given dataset
        # -> X.shape[1] returns the amount of columns from wich we select between 1 and all. Replace false prevents duplicate selections
        features = np.random.choice(X.shape[1], np.random.randint(1, X.shape[1]), replace=False)
        # returns all rows of every column whose index is among the randomly selected
        return X[:, features]
    
class RandomForestTreeRegressor(DecisionTreeRegressor):
    def __init__(self, max_depth, num_thresholds) -> None:
        super().__init__(max_depth, num_thresholds)
    
    # same function as in the Classifier
    def _modify_split_data(self, X): 
        features = np.random.choice(X.shape[1], np.random.randint(1, X.shape[1]), replace=False)
        return X[:, features]
    
class RandomForest:
    def __init__(self) -> None:
        pass
    
class RandomForestClassifier(RandomForest):
    def __init__(self) -> None:
        pass
    
class RandomForestRegressor(RandomForest):
    def __init__(self) -> None:
        pass
    