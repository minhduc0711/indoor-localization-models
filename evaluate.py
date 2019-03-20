import json
import tabulate
from utils import load_data

import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.externals import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    diff = np.abs(y_test - y_pred)

    max_delta_x = np.max(diff[:, 0])
    max_delta_y = np.max(diff[:, 1])

    mean_delta_x = np.mean(diff[:, 0])
    mean_delta_y = np.mean(diff[:, 1])

    avg_dist_err = np.mean(np.sqrt(np.sum(np.power(y_pred-y_test, 2), 1)))

    return [mse, mae, max_delta_x, max_delta_y, mean_delta_x, mean_delta_y, avg_dist_err]


with open("data/cleaned_idx_dict.json") as f:
    train_idx_dict = json.load(f)
X_test, y_test, _ = load_data("./data/hallway_test_feb_25.csv", train_idx_dict)

model_list = []
models = ["poly", "knn", "dtr", "svr"]
for model_name in models:
    fname_model_x = model_name + "_predict_x.joblib"
    fname_model_y = model_name + "_predict_y.joblib"
    model_x = joblib.load("models/joblib/" + fname_model_x)
    model_y = joblib.load("models/joblib/" + fname_model_y)
    model_list.append((model_x, model_y))

columns = ["Model", "MSE", "MAE", "Max x error", "Max y error",
           "Mean x error", "Mean y error", "Average distance error"]
df = pd.DataFrame(columns=columns)
for i, (model_x, model_y) in enumerate(model_list):
    y_pred = np.hstack((model_x.predict(X_test).reshape(-1, 1),
                        model_y.predict(X_test).reshape(-1, 1)))
    model_name = model_x.__class__.__name__

    results = [model_name]
    results.extend(calculate_metrics(y_test, y_pred))
    df.loc[i] = results

nn_model = load_model("models/neural_net.h5")
y_pred = nn_model.predict(X_test)
results = ["Neural net"]
results.extend(calculate_metrics(y_test, y_pred))
df.loc[4] = results

print(tabulate.tabulate(df.values, df.columns, tablefmt="pipe"))
