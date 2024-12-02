import multiprocessing
from functools import partial
import numpy as np

# Scikit-learns GridSearchCV implementation requires the models to inherit from sklearn.base.BaseEstimator,
# which contradicts the requirement of creating the models from scratch.
# => A custom implementation of GridSearchCV is necessary

class GridSearchCV:
    """Grid search with cross validation for hyperparameter optimization"""
    def __init__(self, model, param_grid, cv=5, n_jobs=-1) -> None:
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count() # uses all available cores if n_jobs is set to -1
        self.best_params = None

    def fit(self, X, Y):
        best_score = -np.inf
        best_params = None
        for params in self.param_grid:
            try:
                # tries to create a model with the given parameters
                model = self.model(**params)
            except TypeError:
                raise(TypeError("Model does not support the given parameters"))
            scores = self._cross_val_score(model, X, Y)
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_params = params
        self.best_score = best_score
        self.best_params = best_params

    def _cross_val_score(self, model, X, Y):
        if self.n_jobs == 1:
            return [self._cross_val_score_single(model, X, Y, i) for i in range(self.cv)]
        else:
            with multiprocessing.Pool(self.n_jobs) as pool:
                # pool.map automatically distributes the work to the available processors and collects the results in a list
                # -> here the work is calling the cross_val_score_single method with the given model, input data and slice index
                func = partial(self._cross_val_score_single, model, X, Y)
                return pool.map(func, range(self.cv))

    def _cross_val_score_single(self, model, X, Y, slice_idx):
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
