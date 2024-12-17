# Implementing Decision Trees, Random Forests and Gradient Boosting Trees from scratch

In this project, I will implement the algorithms behind Decision Trees, Random Forests and Gradient Boosting Trees from scratch, using nothing but `numpy`. In examples the performance of my implementations will be tested against a variety of datasets and ultimately compared to the performance of `scikit-learn` library's models for the same datasets.

## How to use

In order to run the examples, make sure the `requirements.txt` are installed in the selected Jupyter Kernel. To do so, run the following command in the terminal:
```bash
pip install -r requirements.txt
```
A similar approach can be used to install the required libraries for any other Python environment. To use the ml-models, simply import them from the models folder. For example:
```python
from models.decision_tree import DecisionTreeClassifier
```
All models inherit the same simple interface from the `BaseEstimator` class that consists of three methods: `fit()`, `predict()` and `score()`. While `fit()` takes the training data and the target labels as arguments and fits the model to the data, `predict()` receives (a set of) samples and returns the models prediction on those values. All models can be tuned using the hyperparameters that are passed as arguments to the constructor. Depending on the model, the hyperparameters may vary, explainations are available as a docstring in each constructor. Ultimately, the `score()` method can be used to evaluate the model's performance on a given dataset and is neccessary to work with the grid search cross validation implementation in the `utils` folder.

## Structure of the project

The project is structured as follows:
```
.
├── README.md
├── examples
│   ├── decision_tree_example.ipynb
│   ├── gradient_boosting_tree_example.ipynb
│   └── random_forest_example.ipynb
├── models
│   ├── base_estimator.py
│   ├── decision_tree.py
│   ├── gradient_boosting_tree.py
│   └── random_forest.py
├── requirements.txt
└── utils
    ├── grid_search_cv.py
    └── reports.py
```
The models folder contains the implementation of the Decision Tree, Random Forest, and Gradient Boosting Tree algorithms. In the examples folder there are practical uses of the models that draw comparisons between the models and the `scikit-learn` implementations. The utils folder contains a custom grid search cross validation implementation that can be used to tune the hyperparameters of the models.

The implementations are all structured in a abstract base class that contains the common methods for regression and classification. The specific methods for each model are implemented in the respective classes that inherit from the base class. All models are suited for both regression and (multiclass-) classification tasks.

## Author
Leon Maag