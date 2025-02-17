import inspect
import copy
from models.base_estimator import BaseEstimator
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

# Scikit-learns GridSearchCV implementation requires the models to inherit from sklearn.base.BaseEstimator,
# which contradicts the requirement of creating the models from scratch.
# => A custom implementation of GridSearchCV is necessary

class GridSearchCV:
    """Grid search with cross validation for hyperparameter optimization"""
    def __init__(self, model: BaseEstimator, param_grid: dict, cv=3, n_jobs=-1) -> None:
        """
        Initializes the GridSearchCV object with the given model, parameter grid, cross-validation strategy, and number of jobs.
        
        Parameters
        ----------
        `model` : BaseEstimator (required)
            The machine learning model class to be optimized
        
        `param_grid` : dict (required)
            Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
            
        `cv` : int, default=5
            Number of folds in cross-validation. Defaults to 5.
            
        `n_jobs` : int, default=-1
            Number of jobs to run in parallel. Defaults to -1, which means using all available cores.
        """
        
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count() # uses all available cores if n_jobs is set to -1
        self.best_params = None
        self.best_model = None
        self.model_supports_parallel = "n_jobs" in inspect.signature(self.model.__init__).parameters.keys() # checks if the model supports parallel processing by checking if the n_jobs parameter is available

    def fit(self, X, Y):
        self.best_score = -np.inf
        self.best_params = None
        
        for params in self.param_grid:
            try:
                if self.model_supports_parallel:
                    params["n_jobs"] = self.n_jobs
                # tries to create a model with the given parameters
                model = self.model(**params)
            except TypeError:
                raise(TypeError("Model does not support the given parameters"))
            scores = self._cross_val_score(model, X, Y)
            score = np.mean(scores)
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        if self.best_params is not None:
            # check if the model supports parallel processing
            if self.model_supports_parallel:
                self.best_params["n_jobs"] = self.n_jobs
                
            self.best_model = self.model(**self.best_params)
            self.best_model.fit(X, Y)
        
    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet")
        return self.best_model.predict(X)

    def _cross_val_score(self, model: BaseEstimator, X, Y):
        # if model itself is suited for parellel processing, we let it handle the parallelization internally
        # otherwise we use the Pool class to distribute the work to the available processors
        if self.n_jobs == 1 or self.model_supports_parallel:
            return [self._cross_val_score_single(copy.deepcopy(model), X, Y, i) for i in range(self.cv)]
        else:
            with Pool(self.n_jobs) as pool:
                # pool.map automatically distributes the work to the available processors and collects the results in a list
                # -> here the work is calling the cross_val_score_single method with the given model, input data and slice index
                func = partial(self._cross_val_score_single, copy.deepcopy(model), X, Y)
                return pool.map(func, range(self.cv))

    def _cross_val_score_single(self, model: BaseEstimator, X, Y, slice_idx):
        # uses the idx to split the data into a training and validation set        
        n_samples = len(X)
        fold_size = n_samples // self.cv
        start = slice_idx * fold_size
        end = start + fold_size if slice_idx != self.cv - 1 else n_samples # avoids potential rounding errors that lead to out of bounds access
        
        X_train = np.delete(X, slice(start, end), axis=0)
        Y_train = np.delete(Y, slice(start, end), axis=0)
        
        X_val = X[start:end]
        Y_val = Y[start:end]
        
        model.fit(X_train, Y_train)
        return model.score(X_val, Y_val)
