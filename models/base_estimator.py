from abc import ABC, abstractmethod
import numpy as np

class BaseEstimator(ABC):
    """ Defines the interface for the machine learning models """
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def fit(self, X, Y) -> None:
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    
    @abstractmethod
    def score(self, X, Y) -> float:
        pass
    
    # Helper functions for calculating the performance of a model
    @staticmethod
    def _calculate_r2(y_true, y_pred):
        """Calculates the r2 score of a model"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    @staticmethod
    def _calculate_accuracy(y_true, y_pred):
        """Calculates the accuracy of a model"""
        return np.sum(y_true == y_pred) / len(y_true)