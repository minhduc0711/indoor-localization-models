
# coding: utf-8

# In[2]:

import json
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

from utils import load_data


# In[3]:

with open("data/cleaned_idx_dict.json") as f:
    train_idx_dict = json.load(f)
X_train_1, y_train_1, _ = load_data(
    "data/hallway_train_dec_27.csv", train_idx_dict)
X_train_2, y_train_2, _ = load_data("data/hallway_train.csv", train_idx_dict)
X_train = np.vstack((X_train_1, X_train_2))
y_train = np.vstack((y_train_1, y_train_2))
X_test, y_test, _ = load_data("data/hallway_test_feb_25.csv", train_idx_dict)

# In[23]:


pipeline_x = PMMLPipeline([('poly', PolynomialFeatures(degree=2)),
                           ('ridge', Ridge(alpha=440))])
pipeline_x.fit(X_train, y_train[:, 0])

pipeline_y = PMMLPipeline([('poly', PolynomialFeatures(degree=2)),
                           ('ridge', Ridge(alpha=440))])
pipeline_y.fit(X_train, y_train[:, 1])


# In[24]:


sklearn2pmml(pipeline_x, "models/pmml/poly_x.pmml", with_repr=True)
sklearn2pmml(pipeline_y, "models/pmml/poly_y.pmml", with_repr=True)
