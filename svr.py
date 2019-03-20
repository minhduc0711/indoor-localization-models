import numpy as np
from sklearn.svm import SVR
from sklearn2pmml.pipeline import PMMLPipeline
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
    ("svr_x", SVR(C=10, cache_size=200, coef0=0.0, degree=1, epsilon=0.5, gamma=0.001,
                  kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False))
])

pipeline_x.fit(X_train, y_train[:, 0])
sklearn2pmml(pipeline_x, "models/pmml/svr_x.pmml", with_repr=True)


pipeline_y = PMMLPipeline([
    ("svr_y", SVR(C=1, cache_size=200, coef0=0.0, degree=1, epsilon=0.5, gamma=0.001,
                  kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False))
])

pipeline_y.fit(X_train, y_train[:, 1])
sklearn2pmml(pipeline_x, "models/pmml/svr_y.pmml", with_repr=True)
