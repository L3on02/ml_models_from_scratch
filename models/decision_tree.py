from abc import ABC, abstractmethod
import numpy as np

class Node:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None
    
    
class DecisionTree(ABC):
    def __init__(self, max_depth) -> None:
        self.tree = None
        self.max_depth = max_depth
        
    def fit(self, X, Y):
        '''creates a decision tree from the data'''
        self.tree = self._build_tree(0, X, Y)
        
    def predict(self, X):
        '''makes a prediction on the input data'''
        return np.array([self._predict(input_data, self.tree) for input_data in X])
        
    def _build_tree(self, depth, X, Y) -> Node:
        # check if max depth is reached -> calculate leaf value for regression or classification (implemented by subclasses)
        # or if the node is pure, return a leaf node with the value/label of that class
        if depth == self.max_depth or len(np.unique(Y)) == 1:
            return Node(value = self._leaf_value(Y))
        
        # best split, implemented by subclasses should return a tuple of featureIndex and threshold where threshold can be a number or a category
        split_feature_index, split_threshold = self._best_split(X, Y)
        
        # if no split is possible, return a leaf node with the value/label of the most common class
        if split_feature_index is None:
            return Node(value = self._leaf_value(Y))
        
        is_numeric = isinstance(split_threshold, (int, float))
        points = X[:, split_feature_index]
        
        # if the feature is not numeric, then split_threshold is a category and we therefore check for equality
        # returns an array of booleans that indicate wether a point should be part of the left split or not
        left_indices = points < split_threshold if is_numeric else points == split_threshold 

        left_X, left_Y = X[left_indices], Y[left_indices]
        # bitwise not for every element in left_indices -> "if not in left it must be in right"
        right_X, right_Y = X[~left_indices], Y[~left_indices]

        # build tree recursively
        left_subtree = self._build_tree(depth=depth+1, X=left_X, Y=left_Y)
        right_subtree = self._build_tree(depth=depth+1, X=right_X, Y=right_Y)
        
        return Node(feature_index=split_feature_index, threshold=split_threshold, left=left_subtree, right=right_subtree)
    
    def _predict(self, input_data, node: Node):
        # if _predict reaches a leave, the prediction has been made and can be returned
        if node.is_leaf():
            return node.value
        
        is_numerical = isinstance(node.threshold, (int, float))
        # differentiation between categorical and numerical features
        if (is_numerical and input_data[node.feature_index] < node.threshold) or (not is_numerical and input_data[node.feature_index] == node.threshold):
            return self._predict(input_data, node.left)
        else:
            return self._predict(input_data, node.right)
        
    def _best_split(self, X, Y):
        best_score = float('inf') # start with worst possible score
        best_feature_index = None
        best_threshold = None
        
        # doesnt modify the data if not overridden -> allows random feature selection for random forest
        features = self._choose_split_indicies(X)
        
        # loop over every for splitting selected feature in X
        for feature_index in features:
            samples = X[:, feature_index]            
            is_numeric = isinstance(X[0, feature_index], (int, float))            
            # now determine possible split thresholds within that feature
            # => split once for every category or along the equally spaced thresholds
            # the amount of thresholds is scaled dynamically to the variance of the data with
            # a minimum of 5 and a maximum of 20 thresholds (or the amount of data points if less)
            unique_values = np.linspace(min(samples), max(samples), min(max(5, int(np.log(np.var(samples) + 1) * 2)), min(len(samples), 20))) if is_numeric else np.unique(samples)
            
            # loop over every possible threshold
            for threshold in unique_values:
                left_indices = samples < threshold if is_numeric else samples == threshold
                
                # split label arrays into respective splits
                left_Y, right_Y = Y[left_indices], Y[~left_indices]
                
                # if the split would result in only one group it is meaningless for the creation of the tree
                # and we dont need to waste resources computing its gini impurity
                if len(left_Y) == 0 or len(right_Y) == 0:
                    continue
                
                score = self._score_split(left_Y, right_Y)
                
                if score < best_score:
                    best_score = score
                    best_feature_index = feature_index
                    best_threshold = threshold
                    
        return best_feature_index, best_threshold 
    
    # can be overridden by subclasses to modify the data before splitting
    def _choose_split_indicies(self, X):
        # be default all indices are considered for splitting
        return [i for i in range(X.shape[1])]

    @abstractmethod
    def _score_split(self, X, Y):
        pass
    
    @staticmethod 
    @abstractmethod
    def _leaf_value(Y):
        pass
    
    
class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth) -> None:
        super().__init__(max_depth)
                        
    # calculates the gini impurity for the classifier
    def _score_split(self, left_Y, right_Y):
        left_sample_count = len(left_Y)
        right_sample_count = len (right_Y)
        total_sample_count = left_sample_count + right_sample_count
        
        left_weight = left_sample_count / total_sample_count
        right_weight = right_sample_count / total_sample_count
        
        # gini is 1 minus the sum of the probability of each classification within its subset squared
        # => c loops over every unique label in Y, the probability of that class is then the sum of all 
        #    occasions where Y is equal to c devided by the total amount of datapoints in Y
        # The whole procedure is done once for the right split and once for the left
        left_gini = 1 - sum((np.sum(left_Y == c) / left_sample_count) ** 2 for c in np.unique(left_Y)) if left_sample_count > 0 else 0
        right_gini = 1 - sum((np.sum(right_Y == c) / right_sample_count) ** 2 for c in np.unique(right_Y)) if right_sample_count > 0 else 0
        
        # averages the right and left gini to return a total evaluation of the split
        return left_weight * left_gini + right_weight * right_gini
    
    # determines most present label using a hashmap to count them
    @staticmethod
    def _leaf_value(Y):
        counts = {}
        for p in Y:
            counts[p] = counts.get(p,0) + 1
        max_count = 0
        label = None
        for l, c in counts.items():
            if c > max_count:
                max_count = c
                label = l
        return label
            
    
class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth) -> None:
        super().__init__(max_depth)
        
    # calculates the Mean Squared Error for the regressor
    def _score_split(self, left_Y, right_Y):
        left_sample_count = len(left_Y)
        right_sample_count = len (right_Y)
        total_sample_count = left_sample_count + right_sample_count
        
        left_weight = left_sample_count / total_sample_count
        right_weight = right_sample_count / total_sample_count
        
        mean_left = np.mean(left_Y)
        mean_right = np.mean(right_Y)
        
        # calculates MSE for each of the two splits. If one split is empty, return inf to indicate bad split
        left_mse = sum((mean_left - l) ** 2 for l in left_Y) / left_sample_count if left_sample_count > 0 else float('inf')
        right_mse = sum((mean_right - r) ** 2 for r in right_Y) / right_sample_count if right_sample_count > 0 else float('inf')
        
        # averages the right and left mse to return a total evaluation of the split
        return left_weight * left_mse + right_weight * right_mse
    
    # returns the average value of all data in the node
    @staticmethod
    def _leaf_value(Y):
        return np.mean(Y)
