import numpy as np
from threading import Thread
from abc import ABC, abstractmethod
from models.decision_tree import DecisionTree, DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth) -> None:
        super().__init__(max_depth)
    
    # method is called in DecisionTree._best_split
    def _modify_split_data(self, X):
        # randomly selects a fixed size (sqrt(total amount of features)) of features to use for the split
        # -> X.shape[1] returns the amount of columns from wich we select between 1 and all. Replace false prevents duplicate selections
        features = np.random.choice(X.shape[1], np.sqrt(X.shape[1]), replace=False)
        # returns all rows of every column whose index is among the randomly selected
        return X[:, features]
    
class RandomForestTreeRegressor(DecisionTreeRegressor):
    def __init__(self, max_depth) -> None:
        super().__init__(max_depth)
    
    # same function as in the Classifier
    def _modify_split_data(self, X):
        features = np.random.choice(X.shape[1], np.sqrt(X.shape[1]), replace=False)
        return X[:, features]
    
class RandomForest(ABC):
    @abstractmethod
    def __init__(self, max_depth, n_samples) -> None:
        self.trees: list[DecisionTree] = []
    
    def fit(self, X, Y):
        '''creates a random forest from the data'''
        threads: list[Thread] = []
        for tree in self.trees:
            # randomizes sample selection and weight for every tree
            # -> some samples will occur multiple times, some not at all (=> replace=True means duplicates are possible)
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            # extracts the rows from the dataset by index which are stored in the indices array
            X_subset, Y_subset = X[indices], Y[indices]
            # since all individual trees are independent, the training can happen in parallel
            thread = Thread(target=tree.fit, args=(X_subset, Y_subset))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            # waits for all threads to finish
            thread.join()
            
    def predict(self, X):
        '''makes a prediction on the input data'''
        return self._predict(X)
        
    @abstractmethod
    def _predict(self, X):
        pass

class RandomForestClassifier(RandomForest):
    def __init__(self, max_depth, n_samples) -> None:
        self.trees = [RandomForestTreeClassifier(max_depth) for _ in range(n_samples)]
        
    def _predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        # returns the most common prediction for every input data
        # -> _leaf_value is already implemented to determine the value of a leaf node
        #    and returns the most common value in a set of rows
        return np.array([RandomForestTreeClassifier._leaf_value(prediction) for prediction in np.array(predictions)])
        
class RandomForestRegressor(RandomForest):
    def __init__(self, max_depth, n_samples) -> None:
        self.trees = [RandomForestTreeRegressor(max_depth) for _ in range(n_samples)]
    
    def _predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        # returns the mean of all predictions for every input data 
        # -> _leaf_value for the regressor returns the mean of all values in a set of rows
        return np.array([RandomForestTreeRegressor._leaf_value(prediction) for prediction in np.array(predictions)])