import numpy as np
from abc import ABC, abstractmethod
from models.decision_tree import DecisionTree, DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth, min_samples_split, min_samples_leaf, num_thresholds) -> None:
        super().__init__(max_depth, min_samples_split, min_samples_leaf, num_thresholds)
    
    # method is called in DecisionTree._best_split
    def _choose_split_indicies(self, X):
        # randomly selects a fixed size (sqrt(total amount of features)) of features to use for the split
        # -> X.shape[1] returns the amount of columns from wich we select between 1 and all. Replace false prevents duplicate selections
        return np.random.choice(X.shape[1], max(1,int(np.sqrt(X.shape[1]))), replace=False)
        
    
class RandomForestTreeRegressor(DecisionTreeRegressor):
    def __init__(self, max_depth, min_samples_split, min_samples_leaf, num_thresholds) -> None:
        super().__init__(max_depth, min_samples_split, min_samples_leaf, num_thresholds)
    
    # same function as in the Classifier
    def _choose_split_indicies(self, X):
        return np.random.choice(X.shape[1], max(1,int(np.sqrt(X.shape[1]))), replace=False)
    
class RandomForest(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.trees: list[DecisionTree] = []
    
    def fit(self, X, Y):
        '''creates a random forest from the data'''
        for tree in self.trees:
            # randomizes sample selection and weight for every tree
            # -> some samples will occur multiple times, some not at all (=> replace=True means duplicates are possible)
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            # extracts the rows from the dataset by index which are stored in the indices array
            X_subset, Y_subset = X[indices], Y[indices]
            
            # this step could happen in parallel, however multithreading is not supported in pythons GIL
            # -> the multiprocessing module created some errors that i have not fixed yet
            tree.fit(X_subset, Y_subset)
            
    def predict(self, X):
        '''makes a prediction on the input data'''
        # collects the predictions of every tree in the forest and transposes the array
        # -> the transposition is necessary to get the predictions for every input data in a single row
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        # returns the mean or most common prediction among each trees prediction for each sample in the input data
        return self._evaluate(predictions)
    
    @staticmethod
    @abstractmethod
    def _evaluate(self, X):
        pass

class RandomForestClassifier(RandomForest):
    def __init__(self, n_samples = 20, max_depth = 15, min_samples_split = 5, min_samples_leaf = 5, num_thresholds = 10) -> None:
        self.trees = [RandomForestTreeClassifier(max_depth, min_samples_split, min_samples_leaf, num_thresholds) for _ in range(n_samples)]
    
    @staticmethod  
    def _evaluate(predictions):
        # -> _leaf_value for the classifier returns the most common label in a set of rows
        return np.array([RandomForestTreeClassifier._leaf_value(prediction) for prediction in predictions])
        
class RandomForestRegressor(RandomForest):
    def __init__(self, n_samples = 20, max_depth = 15, min_samples_split = 5, min_samples_leaf = 5, num_thresholds = 10) -> None:
        self.trees = [RandomForestTreeRegressor(max_depth, min_samples_split, min_samples_leaf, num_thresholds) for _ in range(n_samples)]
    
    @staticmethod  
    def _evaluate(predictions):
        # -> _leaf_value for the regressor returns the mean of all values in a set of rows
        return np.array([RandomForestTreeRegressor._leaf_value(prediction) for prediction in predictions])