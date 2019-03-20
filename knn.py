import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from utils import load_data
import json

with open("data/cleaned_idx_dict.json") as f:
    train_idx_dict = json.load(f)

X_train_1, y_train_1, _ = load_data("data/hallway_train_dec_27.csv", train_idx_dict)
X_train_2, y_train_2, _ = load_data("data/hallway_train.csv", train_idx_dict)
X_train = np.vstack((X_train_1, X_train_2))
y_train = np.vstack((y_train_1, y_train_2))
X_test, y_test, _ = load_data("data/hallway_test_feb_25.csv", train_idx_dict)

pipeline_x = PMMLPipeline([
    ("knr_x", KNeighborsRegressor(algorithm='auto', leaf_size=3,
                                  metric='minkowski', n_jobs=1, n_neighbors=5, p=2, weights='uniform'))
])

pipeline_x.fit(X_train, y_train[:, 0])
sklearn2pmml(pipeline_x, "models/pmml/knn_x.pmml", with_repr=True)


pipeline_y = PMMLPipeline([
    ("knr_y", KNeighborsRegressor(algorithm='brute', leaf_size=6, metric='minkowski',
                                  metric_params=None, n_jobs=1, n_neighbors=11, p=2,
                                  weights='uniform'))
])

pipeline_y.fit(X_train, y_train[:, 1])
sklearn2pmml(pipeline_x, "models/pmml/knn_y.pmml", with_repr=True)
