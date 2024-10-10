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
        self.tree = self._build_tree(X, Y)
        
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


    @abstractmethod
    def _best_split(self, X, Y):
        pass
        
    @abstractmethod
    def _leaf_value(self, Y):
        pass
    
class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth, num_thresholds) -> None:
        DecisionTree.__init__(max_depth)
        self.num_thresholds = num_thresholds
        
    def _best_split(self, X, Y):
        best_gini = float(1.0) # start with worst possible gini impurity (actually 1 is not even possible)
        best_feature_index = None
        best_threshold = None
        
        # loop over every feature in X
        for feature_index in range(X.shape[1]):
            points = X[:, feature_index]
            is_numeric = isinstance(X[0, feature_index], (int, float))
            # now determine possible split thresholds within that feature
            # => split once for every category or along equally spaced thresholds
            unique_values = np.linspace(min(points), max(points), self.num_thresholds) if is_numeric else np.unique(points)
            
            # loop over every possible threshold
            for threshold in unique_values:
                left_indices = points < threshold if is_numeric else points == threshold
                
                # split label arrays into respective splits
                left_Y, right_Y = Y[left_indices], Y[~left_indices]
                
                # if the split would result in only one group it is meaningless for the creation of the tree
                # and we dont need to waste resources computing its gini impurity
                if len(left_Y) == 0 or len(right_Y) == 0:
                    continue
                
                gini = self._calc_gini(left_Y, right_Y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold
                    
        return best_feature_index, best_threshold 
                        
    
    @staticmethod
    def _calc_gini(left_Y, right_Y):
        left_sample_count = len(left_Y)
        right_sample_count = len (right_Y)
        total_sample_count = left_sample_count + right_sample_count
        
        left_weight = len(left_Y) / total_sample_count
        right_weight = len(right_weight) / total_sample_count
        
        # gini is 1 minus the sum of the probability of each classification within its subset squared
        # => c loops over every unique label in Y, the probability of that class is then the sum of all 
        #    occasions where Y is equal to c devided by the total amount of datapoints in Y
        # The whole procedure is done once for the right split and once for the left
        left_gini = 1 - sum((np.sum(left_Y == c) / left_sample_count) ** 2 for c in np.unique(left_Y)) if left_sample_count > 0 else 0
        right_gini = 1 - sum((np.sum(right_Y == c) / right_sample_count) ** 2 for c in np.unique(right_Y)) if right_sample_count > 0 else 0
        
        # averages the right and left gini to return a total evaluation of the split
        return left_weight * left_gini + right_weight * right_gini
    
class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth) -> None:
        DecisionTree.__init__(max_depth)
    
    def _best_split(self, X, Y):
        pass
    
    @staticmethod
    def calc_mse(left_Y, right_y):
        pass