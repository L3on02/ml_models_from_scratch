import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from models.decision_tree import DecisionTreeRegressor

class GradientBoostingTree(ABC):
    def __init__(self, n_estimators, learning_rate, patience, max_depth, min_samples_split, min_samples_leaf, num_thresholds) -> None:
        self.estimators: list[DecisionTreeRegressor] = []
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_thresholds = num_thresholds
        
    def fit(self, X, Y):
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)
        
        # the GBT prediction needs to be initialized with a starting value,
        # in the case of regression we use the mean of the target values
        # in the case of classification the log(odds)
        self.initial_prediction = self._initial_prediction(Y_train)
        
        # initialize the entire array with the initial prediction
        predictions = np.full(Y_train.shape, self.initial_prediction)
        best_score = float('inf')
        no_improvement_rounds = 0
        
        # consecutively build trees onto one another
        for _ in range(self.n_estimators):
            # first the residuals are computed as the negative gradient of the loss function which,
            # depending on whether the model is a classifier or regressor, is
            # MSE or LogLoss
            residuals = self._compute_residuals(Y_train, predictions)
            # create a new weak learner (DecisionTree) and fit it to train data
            # but use the residuals as the target variable
            # weak learner is always a Regressor, eventhough the ensamble might be used for
            # classification. It predicts the residuals which are the error of the previous prediction
            estimator = DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.num_thresholds)
            estimator.fit(X_train, residuals)
            
            # calculate the mean squared error of the validation set
            score = self._evaluate(X_validation, Y_validation)
            # check if score has improved compared to the last iterations and prematurely stop the training
            # when no improvement is made for self.patience amount of rounds
            # => prevents overfitting and improves performance
            if score >= best_score:
                no_improvement_rounds += 1
                if no_improvement_rounds >= self.patience:
                    break
            else:
                best_score = score
                no_improvement_rounds = 0
            
            # update the predictions with the new estimator multiplied by the learning rate
            predictions += self.learning_rate * estimator.predict(X_train)
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
    def _evaluate(self, X, Y):
        pass
    
    @abstractmethod
    def _initial_prediction(self, Y):
        pass
    
    @abstractmethod
    def _interpret_prediction(self, prediction):
        pass
    
class GradientBoostingClassifier(GradientBoostingTree):
    def __init__(self, n_estimators = 50, learning_rate = 0.1, patience = 10, max_depth = 10, min_samples_split = 5, min_samples_leaf = 5, num_thresholds = 10) -> None:
        super().__init__(n_estimators, learning_rate, patience, max_depth, min_samples_split, min_samples_leaf, num_thresholds)     
        
    def _compute_residuals(self, Y, prediction):
        # uses the sigmoid function to turn the predicted value into a probability
        # of the sample belonging to the positive class
        # which is then subtracted from the actual label to get the residual
        return Y - (1 / (1 + np.exp(-prediction)))
    
    def _initial_prediction(self, Y):
        # the initial prediction for the classifier is the log(odds) of proportion of positive classes over all classes
        p = np.clip(np.mean(Y), 1e-10, 1 - 1e-10) # since the formula devides through p, we clip out 0 and 1 to avoid division by zero
        return np.log(p / (1 - p))
    
    def _interpret_prediction(self, prediction):
        # converts the prediction to a probability and then rounds it to 0 or 1
        # with the threshold being 0.5
        return np.round(1 / (1 + np.exp(-prediction)), 0)
    
    def _evaluate(self, X, Y):
        # calculates the log loss of the predictions
        predictions = np.clip(self.predict(X), 1e-10, 1 - 1e-10) # clip to avoid log(0) or log(1)
        return -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
    
    # relevant for multi-class classification
    def _one_hot_encode(self, Y):
        # returns all unique labels
        classes = np.unique(Y)
        # then create a vector for each row in Y that has len(classes) elements and a "True" at the index of its label
        # the array needs to be transposed to have the shape (samples, one hot encoded classes)
        return np.array([Y == c for c in classes]).T  
    
class GradientBoostingRegressor(GradientBoostingTree):
    def __init__(self, n_estimators = 50, learning_rate = 0.1, patience = 10, max_depth = 10, min_samples_split = 5, min_samples_leaf = 5, num_thresholds = 10) -> None:
        super().__init__(n_estimators, learning_rate, patience, max_depth, min_samples_split, min_samples_leaf, num_thresholds)  
        
    def _compute_residuals(self, Y, prediction):
        # the derivative of the MSE loss simply is the difference between the target and the prediction
        return Y - prediction
    
    def _initial_prediction(self, Y):
        # a reasonable way to determine the initial prediction is use the mean of the target values
        return np.mean(Y)
    
    def _interpret_prediction(self, prediction):
        # for regression the prediction can be returned "as is"
        return prediction
    
    def _evaluate(self, X, Y):
        # calculates the mean squared error of the predictions
        return np.mean((Y - self.predict(X)) ** 2)
