import numpy as np
from abc import ABC, abstractmethod
from models.decision_tree import DecisionTreeRegressor

class GradientBoostingTree(ABC):
    def __init__(self, n_estimators, max_depth, learning_rate) -> None:
        self.estimators: list[DecisionTreeRegressor] = []
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
    def fit(self, X, Y):
        # the GBT prediction needs to be initialized with a starting value,
        # in the case of regression we use the mean of the target values
        # in the case of classification the log(odds)
        self.initial_prediction = self._initial_prediction(Y)
        
        # initialize the entire array with the initial prediction
        predictions = np.full(Y.shape, self.initial_prediction)
        
        # consecutively build trees onto one another
        for _ in range(self.n_estimators):
            # first the residuals are computed as the negative gradient of the loss function which,
            # depending on whether the model is a classifier or regressor, is
            # MSE or cross-entropy loss
            residuals = self._compute_residuals(Y, predictions)
            # create a new weak learner (DecisionTree) and fit it to train data
            # but use the residuals as the target variable
            # weak learner is always a Regressor, eventhough the ensamble might be used for
            # classification. It predicts the residuals which are the error of the previous prediction
            estimator = DecisionTreeRegressor(max_depth=self.max_depth)
            estimator.fit(X, residuals)
            # update the predictions with the new estimator multiplied by the learning rate
            predictions += self.learning_rate * estimator.predict(X)
            # append the new estimator to the existing ones in the ensemble
            self.estimators.append(estimator)
            
    def predict(self, X):
        # initialize every prediction with the original initial_prediction made for the first estimator during fitting
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # now apply the prediction of every estimator in the ensemble to each prediction
        for estimator in self.estimators:
            predictions += self.learning_rate * estimator.predict(X)
            
        # for Regression the predictions from the estimator can be returned "as is",
        # for classification the predictions need to be interpreted since the predictions
        # represent probabilities
        return np.array([self._interpret_prediction(prediction) for prediction in predictions])
    
    @abstractmethod
    def _compute_residuals(self, Y, prediction):
        pass
    
    @abstractmethod
    def _initial_prediction(self, Y):
        pass
    
    @abstractmethod
    def _interpret_prediction(self, prediction):
        pass
    
class GradientBoostingClassifier(GradientBoostingTree):
    def __init__(self, n_estimators, max_depth, learning_rate) -> None:
        super().__init__(n_estimators, max_depth, learning_rate)       
    
class GradientBoostingRegressor(GradientBoostingTree):
    def __init__(self, n_estimators, max_depth, learning_rate) -> None:
        super().__init__(n_estimators, max_depth, learning_rate)
        
    def _compute_residuals(self, Y, prediction):
        # the derivative of the MSE loss simply is the difference between the target and the prediction
        return Y - prediction
    
    def _initial_prediction(self, Y):
        # a reasonable way to determine the initial prediction is use the mean of the target values
        return np.mean(Y)
    
    def _interpret_prediction(self, prediction):
        # for regression the prediction can be returned "as is"
        return prediction