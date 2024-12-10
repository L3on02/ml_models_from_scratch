from models.base_estimator import BaseEstimator
from models.decision_tree import DecisionTree, DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from abc import abstractmethod
from multiprocessing import Pool, cpu_count

class RandomForestTreeClassifier(DecisionTreeClassifier):
    def __init__(self, rng, max_depth, min_samples_split, min_samples_leaf, num_thresholds) -> None:
        super().__init__(max_depth, min_samples_split, min_samples_leaf, num_thresholds)
        self.rng = rng
    
    # method is called in DecisionTree._best_split
    def _choose_split_indicies(self, X):
        # randomly selects a fixed size (sqrt(total amount of features)) of features to use for the split
        # -> X.shape[1] returns the amount of columns from wich we select between 1 and all. Replace false prevents duplicate selections
        return np.random.RandomState(42).choice(X.shape[1], max(1,int(np.sqrt(X.shape[1]))), replace=False)
        
    
class RandomForestTreeRegressor(DecisionTreeRegressor):
    def __init__(self, rng, max_depth, min_samples_split, min_samples_leaf, num_thresholds) -> None:
        super().__init__(max_depth, min_samples_split, min_samples_leaf, num_thresholds)
        self.rng= rng
    
    # same function as in the Classifier
    def _choose_split_indicies(self, X):
        return self.rng.choice(X.shape[1], max(1,int(np.sqrt(X.shape[1]))), replace=False)
    
    
class RandomForest(BaseEstimator):
    @abstractmethod
    def __init__(self, n_jobs) -> None:
        self.trees: list[DecisionTree] = []
        self.rng = None
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1,min(n_jobs, cpu_count()))
    
    def fit(self, X, Y):
        """creates a random forest from the data"""
        # randomizes sample selection and weight for every tree
        # -> some samples will occur multiple times, some not at all (=> replace=True means duplicates are possible)
        # extracts the rows from the dataset by index which are stored in the indices array
        if self.n_jobs == 1:
            for idx, tree in enumerate(self.trees):
                self._fit_estimator((tree, X, Y, idx))
        else:
            with Pool(self.n_jobs) as pool:
                self.trees = pool.map(self._fit_estimator, [(tree, X, Y, idx) for idx, tree in enumerate(self.trees)])        
        
    def predict(self, X):
        """makes a prediction on the input data"""
        # collects the predictions of every tree in the forest and transposes the array
        # -> the transposition is necessary to get the predictions for every input data in a single row
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        # returns the mean or most common prediction among each trees prediction for each sample in the input data
        return self._evaluate(predictions)    
    
    def _fit_estimator(self, args: tuple[DecisionTree, np.ndarray, np.ndarray, int]):
        tree, X, Y, seed = args
        # every tree gets its own random number generator with a unique seed to ensure different samples
        rng = np.random.default_rng(seed)
        # extracts the rows from the dataset by index which are stored in the indices array
        indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        X_subset, Y_subset = X[indices], Y[indices]
        tree.fit(X_subset, Y_subset)
        return tree
    
    @abstractmethod
    def _evaluate(self, X):
        pass       


class RandomForestClassifier(RandomForest):
    def __init__(self, n_jobs = -1, n_estimators = 20, random_seed = 42, max_depth = 15, min_samples_split = 5, min_samples_leaf = 5, num_thresholds = 10) -> None:
        """A random forrest classifier that uses decision trees as weak learners.

        Parameters
        ----------
        `n_jobs` : int, default=-1
            The number of jobs to run in parallel. -1 means using all available processors.
        
        `n_estimators` : int, default=20
            The number of weak learners in the ensemble.
            
        `random_seed` : int, default=42
            The seed used for the random number generator.
        
        Weak learner parameters:
        
        `max_depth` : int, default=15
            The maximum depth of the tree, when no other stopping criteria are met.

        `min_samples_split` : int, default=5
            The minimum number of samples required to split an internal node.

        `min_samples_leaf` : int, default=5
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least
            min_samples_leaf training samples in each of the left and right branches.

        `num_thresholds` : int, default=10
            The number of thresholds to consider when finding the best split
            for a numeric feature.
        """
        super().__init__(n_jobs)
        self.rng = np.random.default_rng(random_seed)
        self.trees = [RandomForestTreeClassifier(self.rng, max_depth, min_samples_split, min_samples_leaf, num_thresholds) for _ in range(n_estimators)]
    
    def score(self, X, Y):
        """calculates the accuracy of the model"""
        Y_pred = self.predict(X)
        return BaseEstimator._calculate_accuracy(Y, Y_pred)
    
    def _evaluate(predictions):
        # -> _leaf_value for the classifier returns the most common label in a set of rows
        return np.array([RandomForestTreeClassifier._leaf_value(prediction) for prediction in predictions])
    
        
class RandomForestRegressor(RandomForest):
    def __init__(self, n_jobs = -1, n_estimators = 20, random_seed = 42, max_depth = 15, min_samples_split = 5, min_samples_leaf = 5, num_thresholds = 10) -> None:
        """A random forrest regressor that uses decision trees as weak learners.

        Parameters
        ----------
        `n_jobs` : int, default=-1
            The number of jobs to run in parallel. -1 means using all available processors.
        
        `n_estimators` : int, default=20
            The number of weak learners in the ensemble.
            
        `random_seed` : int, default=42
            The seed used for the random number generator.
        
        Weak learner parameters:
        
        `max_depth` : int, default=15
            The maximum depth of the tree, when no other stopping criteria are met.

        `min_samples_split` : int, default=5
            The minimum number of samples required to split an internal node.

        `min_samples_leaf` : int, default=5
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least
            min_samples_leaf training samples in each of the left and right branches.

        `num_thresholds` : int, default=10
            The number of thresholds to consider when finding the best split
            for a numeric feature.
        """
        super().__init__(n_jobs)
        self.rng = np.random.default_rng(random_seed)
        self.trees = [RandomForestTreeRegressor(self.rng, max_depth, min_samples_split, min_samples_leaf, num_thresholds) for _ in range(n_estimators)]
   
    def score(self, X, Y):
        """calculates the r2 score of the model"""
        Y_pred = self.predict(X)
        return BaseEstimator._calculate_r2(Y, Y_pred)
    
    def _evaluate(predictions):
        # -> _leaf_value for the regressor returns the mean of all values in a set of rows
        return np.array([RandomForestTreeRegressor._leaf_value(prediction) for prediction in predictions])
