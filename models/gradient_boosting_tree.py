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
        # preprocess the data if necessary
        X, Y = self._preprocess(X, Y)
                
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
    
    def _preprocess(self, X, Y):
        return X, Y
    
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
    
class GradientBoostingClassifier: #(GradientBoostingTree):
    def __init__(self, n_estimators = 50, learning_rate = 0.1, patience = 10, max_depth = 10, min_samples_split = 5, min_samples_leaf = 5, num_thresholds = 10) -> None:
        #super().__init__(n_estimators, learning_rate, patience, max_depth, min_samples_split, min_samples_leaf, num_thresholds)     
        self.estimators: list[DecisionTreeRegressor] = []
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.num_thresholds = num_thresholds
        
    def fit(self, X, Y):
        Y_one_hot = self._one_hot_encode(Y)
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)
        
        n_classes = Y_train.shape[1]
        
        self.initial_prediction = np.log(1 / Y_train.shape[1])
        # fills a 2D array with the same initial prediction for each class
        predictions = np.full(Y_train.shape, self.initial_prediction)
        
        self.estimators = []
        
        best_score = float('inf')
        no_improvement_rounds = 0
        
        # consecutively build trees onto one another
        for _ in range(self.n_estimators):
            class_estimators = []
            new_predictions = np.zeros_like(predictions)
            
            softmax_probabilities = self._softmax(predictions)
            
            for class_idx in range(n_classes):
                residuals = Y_train[:, class_idx] - softmax_probabilities[:, class_idx]
            
                estimator = DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.num_thresholds)
                estimator.fit(X_train, residuals)
                class_estimators.append(estimator)
                
                new_predictions[:, class_idx] = self.learning_rate * estimator.predict(X_train)
                
            predictions += new_predictions
            
            self.estimators.append(class_estimators)
            
            score = self._evaluate(X_validation, Y_validation)
            
            # since the loss function is the cross entropy loss, its scores are naturally higher when there are more classes
            # -> we scale the early stopping criterion by the number of classes
            if best_score - score <= 0.01 * Y_validation.shape[1]:
                no_improvement_rounds += 1
                if no_improvement_rounds >= self.patience:
                    break
            else:
                best_score = score
                no_improvement_rounds = 0

            
    def predict(self, X):
        return np.argmax(self._predict_probability(X), axis=1)
    
    def _predict_probability(self, X):
        predictions = np.full((X.shape[0], len(self.estimators[0])), self.initial_prediction)
        
        for class_estimators in self.estimators:
            for class_idx, estimator in enumerate(class_estimators):
                predictions[:, class_idx] += self.learning_rate * estimator.predict(X)
                
        return self._softmax(predictions)
 
    def _initial_prediction(self, Y):
        return np.log(1 / Y.shape[1])
    
    def _preprocess(self, X, Y):
        return X, self._one_hot_encode(Y)
 
    def _softmax(self, predictions):
        # calculates the exponential of the predictions 
        # the subtraction of the maximum value in the array is done to avoid potential overflows 
        exp = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        # then the exponential is divided by the sum of all exponentials in the row
        # since keepdims=True the division will be applied to each element individually
        # now each row will sum up to 1
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def _evaluate(self, X, Y_oh):
        # calculates the cross entropy loss of the predictions
        probabilities = self._predict_probability(X)
        return -np.mean(np.sum(Y_oh * np.log(np.clip(probabilities, 1e-10, 1 - 1e-10)), axis=1))
        
    
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
