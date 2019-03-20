import numpy as np

from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml import sklearn2pmml
from utils import load_data
import json

with open("data/cleaned_idx_dict.json") as f:
    train_idx_dict = json.load(f)
X_train_1, y_train_1, _ = load_data(
    "data/hallway_train_dec_27.csv", train_idx_dict)
X_train_2, y_train_2, _ = load_data("data/hallway_train.csv", train_idx_dict)
X_train = np.vstack((X_train_1, X_train_2))
y_train = np.vstack((y_train_1, y_train_2))
X_test, y_test, _ = load_data("data/hallway_test_feb_25.csv", train_idx_dict)

pipeline_x = PMMLPipeline([
    ("tree_x",
     DecisionTreeRegressor(criterion='friedman_mse', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=9, min_samples_split=2,
                           min_weight_fraction_leaf=0.02, presort=False, random_state=None,
                           splitter='best'))
])

pipeline_x.fit(X_train, y_train[:, 0])
sklearn2pmml(pipeline_x, "models/pmml/decision_tree_x.pmml", with_repr=True)


pipeline_y = PMMLPipeline([
    ("tree_y", DecisionTreeRegressor(criterion='friedman_mse', max_depth=None,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=9, min_samples_split=2,
                                     min_weight_fraction_leaf=0.02, presort=False, random_state=None,
                                     splitter='best'))
])

pipeline_y.fit(X_train, y_train[:, 1])
sklearn2pmml(pipeline_x, "models/pmml/decision_tree_y.pmml", with_repr=True)
