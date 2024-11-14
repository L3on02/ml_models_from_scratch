# Implementing Decision Trees, Random Forests, and Gradient Boosting Trees from scratch

In this project, I will implement the basic algorithms for Decision Trees, Random Forests, and Gradient Boosting Trees from scratch, using nothing but NumPy and pandas. Using the sklearn library, I will compare the results of my implementations with those of the library to evaluate their performance.

## How to use

In order to run the examples, make sure the `requirements.txt` are installed in the selected Jupyter Kernel. To do so, run the following command in the terminal:
```bash
pip install -r requirements.txt
```
A similar approach can be used to install the required libraries for any other Python environment. To use the ml-models, simply import them from the models folder. For example:
```python
from models.decision_tree import DecisionTreeClassifier
```
All models have the same simple interface that consits of the two methods `fit` and `predict`. The `fit` method takes the training data and the target values, while the `predict` method receives the test data and returns the predicted values. The models can be tuned using the hyperparameters that are passed as arguments to the constructor. Depending on the model, the hyperparameters may vary, explainations are available as a docstring in each constructor.

## Structure of the project

The project is structured as follows:
```
.
├── README.md
├── examples
│   ├── decision_tree_example.py
│   ├── gradient_boosting_tree_example.py
│   └── random_forest_example.py
├── models
│   ├── __init__.py
│   ├── decision_tree.py
│   ├── gradient_boosting_tree.py
│   └── random_forest.py
└── requirements.txt
```
The models folder contains the implementation of the Decision Tree, Random Forest, and Gradient Boosting Tree algorithms. In the examples folder there are practical uses of the models. There also are comparables with the sklearn library.

The implementations are all structured in a abstract base class that contains the common methods for regression and classification. The specific methods for each model are implemented in the respective classes that inherit from the base class. All models are suited for both regression and (multiclass-) classification tasks.