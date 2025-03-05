from models.base_estimator import BaseEstimator
from models.decision_tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

class GradientBoostingTree(BaseEstimator):
    def __init__(self, n_estimators, learning_rate, patience, tolerance, random_state, max_depth, min_samples_split, min_samples_leaf, num_thresholds) -> None:
        self.estimators = []
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.patience = patience
        self.tolerance = tolerance
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_thresholds = num_thresholds
        
    def fit(self, X, Y):
        """creates a gradient boosting tree from the data"""
        # preprocess the data if necessary
        Y = self._preprocess(Y)
                
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=self.random_state)
        
        # the GBT prediction needs to be initialized with a starting value,
        # in the case of regression we use the mean of the target values
        # in the case of classification the log(odds)   
        self.initial_prediction = self._initial_prediction(Y_train)
        
        # initialize the entire array with the initial prediction
        predictions = np.full(Y_train.shape, self.initial_prediction)
        best_score = float('inf')
        no_improvement_rounds = 0
        
        # consecutively build trees onto one another
        for iter in range(self.n_estimators):
            
            estimator, predictions = self._train_iteration(X_train, Y_train, predictions)
            
            self.estimators.append(estimator)
            
            # calculate a score for the current state of the model based on the validation data
            score = self._evaluate(X_validation, Y_validation)
            
            # check if score has improved by at least 'tolerance' within the last 'patience' rounds
            # we prematurely stop the training => prevents overfitting and improves performance
            if best_score - score <= self.tolerance:
                no_improvement_rounds += 1
                if no_improvement_rounds >= self.patience:
                    self.n_iterations = iter
                    break
            else:
                best_score = score
                no_improvement_rounds = 0
                
    def predict(self, X):
        """makes a prediction on the input data"""
        return self._predict(X)

    @abstractmethod 
    def _preprocess(self, Y):
        pass
    
    @abstractmethod       
    def _predict(self, X):
        pass
    
    @abstractmethod
    def _evaluate(self, X, Y):
        pass
    
    @abstractmethod
    def _initial_prediction(self, Y):
        pass
    
    @abstractmethod
    def _train_iteration(self, X, Y, predictions):
        pass
    
class GradientBoostingClassifier(GradientBoostingTree):
    def __init__(self, n_estimators=100, learning_rate=0.15, patience=5, tolerance=1e-4, random_state=41, max_depth=10, min_samples_split=5, min_samples_leaf=5, num_thresholds=10) -> None:
        """A gradient boosting classifier suited for multiclass classification tasks that uses decision tree regressors as weak learners.

        Parameters
        ----------
        `n_estimators` : int, default=50
            The maximum number of weak learners in the ensemble.
            
        `learning_rate` : float, default=0.1
            The learning rate shrinks the contribution of each weak learner.
            
        `patience` : int, default=10
            The number of rounds without improvement before the training is stopped early.
            
        `tolerance` : float, default=1e-4, bounds=[0, inf)
            The minimum improvement in the score over *patience* rounds to be considered as an improvement.
            
        `random_state` : int, default=42
            The seed used by the random number generator for the train-test split.
        
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
        self.estimators: list[list[DecisionTreeRegressor]] = []
        super().__init__(n_estimators, learning_rate, patience, tolerance, random_state, max_depth, min_samples_split, min_samples_leaf, num_thresholds)
    
    def score(self, X, Y):
        """calculates the accuracy of the model"""
        Y_pred = self.predict(X)
        return self._calculate_accuracy(Y, Y_pred)
    
    def _train_iteration(self, X, Y, predictions):
        iteration_estimator = []
        n_classes = Y.shape[1]
        new_predictions = np.zeros_like(predictions)
            
        # scales all predictions to probabilities across classes that sum up to 1
        softmax_probabilities = self._softmax(predictions)
        
        # now train one estimator for each class
        for class_idx in range(n_classes):
            # the residuals are the difference between the target (one hot encoded to 0 or 1) and the predicted probability
            residuals = Y[:, class_idx] - softmax_probabilities[:, class_idx]

            # create a new weak learner (DecisionTree) and fit it to new residuals
            estimator = DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.num_thresholds)
            estimator.fit(X, residuals)
            iteration_estimator.append(estimator)
            
            # since the predictions are used to calculate the residuals, they need to be updated all at once after all estimators are trained
            new_predictions[:, class_idx] = predictions[:, class_idx] + self.learning_rate * estimator.predict(X)
        
        return iteration_estimator, new_predictions
       
    def _predict(self, X):
        # returns the class with the highest probability across all classes
        return np.argmax(self._predict_probability(X), axis=1)
    
    def _preprocess(self, Y):
        # for (multi) classification the target variable Y needs to be one-hot encoded
        classes = np.unique(Y)
        # then create a vector for each row in Y that has len(classes) elements and a "True" at the index of its label
        # the array needs to be transposed to have the shape (samples, one hot encoded classes)
        return np.array([Y == c for c in classes]).T  
    
    def _initial_prediction(self, Y):
        return np.log(1 / Y.shape[1])
    
    def _evaluate(self, X, Y):
        # calculates the cross entropy loss of the predictions
        probabilities = self._predict_probability(X)
        return -np.mean(np.sum(Y * np.log(np.clip(probabilities, 1e-10, 1 - 1e-10)), axis=1))
    
    def _predict_probability(self, X):
        predictions = np.full((X.shape[0], len(self.estimators[0])), self.initial_prediction)
        
        for class_estimators in self.estimators:
            for class_idx, estimator in enumerate(class_estimators):
                # the predictions are updated for each class individually
                predictions[:, class_idx] += self.learning_rate * estimator.predict(X)
        
        # since the predictions can be any value, they need to be scaled to probabilities using the softmax function
        return self._softmax(predictions)
 
    def _softmax(self, predictions):
        # calculates the exponential of the predictions 
        # the subtraction of the maximum value in the array is done to avoid potential overflows 
        exp = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        # then the exponential is divided by the sum of all exponentials in the row
        # since keepdims=True the division will be applied to each element individually
        # now each row will sum up to 1
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    
class GradientBoostingRegressor(GradientBoostingTree):
    def __init__(self, n_estimators=100, learning_rate=0.15, patience=5, tolerance=1e-4, random_state=41, max_depth=10, min_samples_split=5, min_samples_leaf=5, num_thresholds=10) -> None:
        """A gradient boosting regressor decision tree regressors as weak learners.

        Parameters
        ----------
        `n_estimators` : int, default=50
            The maximum number of weak learners in the ensemble.
            
        `learning_rate` : float, default=0.1
            The learning rate shrinks the contribution of each weak learner.
            
        `patience` : int, default=10
            The number of rounds without improvement before the training is stopped early.
            
        `tolerance` : float, default=1e-4, bounds=[0, inf)
            The minimum improvement in the score over *patience* rounds to be considered as an improvement.
        
        `random_state` : int, default=42
            The seed used by the random number generator for the train-test split.
        
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
        self.estimators: list[DecisionTreeRegressor] = []
        super().__init__(n_estimators, learning_rate, patience, tolerance, random_state, max_depth, min_samples_split, min_samples_leaf, num_thresholds)  

    def score(self, X, Y):
        """calculates the r2 score of the model"""
        Y_pred = self.predict(X)
        return self._calculate_r2(Y, Y_pred)
    
    def _train_iteration(self, X, Y, predictions):
        # first the residuals are computed as the negative gradient of the loss function
        # => the derivative of the MSE loss simply is the difference between the target and the prediction
        residuals = Y - predictions
        
        # create a new weak learner (DecisionTree) and fit it to train data
        # but use the residuals as the target variable
        # weak learner is always a Regressor, eventhough the ensamble might be used for
        # classification. It predicts the residuals which are the error of the previous prediction
        iteration_estimator = DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.num_thresholds)
        iteration_estimator.fit(X, residuals)
        
        # update the predictions with the new estimator multiplied by the learning rate
        new_predictions = predictions + self.learning_rate * iteration_estimator.predict(X)
        return iteration_estimator, new_predictions
    
    def _predict(self, X):  
        # initialize every prediction with the original initial_prediction made for the first estimator during fitting
        predictions = np.full(X.shape[0], self.initial_prediction)
        # now apply the prediction of every estimator in the ensemble to each prediction
        for estimator in self.estimators:
            predictions += self.learning_rate * estimator.predict(X)
            
        # for Regression the predictions from the estimator can be returned "as is",
        return predictions
    
    def _preprocess(self, Y):
        # no preprocessing necessary for regression
        return Y
    
    def _initial_prediction(self, Y):
        # a reasonable way to determine the initial prediction is use the mean of the target values
        return np.mean(Y)
    
    def _evaluate(self, X, Y):
        Y_pred = self.predict(X)
    
        # Estimate variance as the mean squared residual (empirical variance)
        variance = np.mean((Y - Y_pred) ** 2)
        variance = max(variance, 1e-6)

        # Formula for the Gaussian Negative Log Likelihood
        return np.mean(0.5 * np.log(2 * np.pi * variance) + ((Y - Y_pred) ** 2) / (2 * variance))
         