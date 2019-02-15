
# coding: utf-8

# In[2]:


import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing as maxabs_scale
from sklearn.preprocessing import *
from matplotlib.pyplot import *
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

from utils import load_data


# In[3]:


X_train, y_train, train_idx_dict = load_data("data/hallway_train.csv")
X_test, y_test, _ = load_data("data/hallway_test.csv", train_idx_dict)

X_train = maxabs_scale(X_train, copy=False)
X_test = maxabs_scale(X_test, copy=False)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[23]:


pipeline_x = PMMLPipeline([('poly', PolynomialFeatures(degree=2)),
                         ('ridge', Ridge(alpha=440))])
pipeline_x.fit(X_train, y_train[:, 0])

pipeline_y = PMMLPipeline([('poly', PolynomialFeatures(degree=2)),
                         ('ridge', Ridge(alpha=440))])
pipeline_y.fit(X_train, y_train[:, 1])


# In[24]:


sklearn2pmml(pipeline_x, "pmml_files/poly_x.pmml", with_repr = True)
sklearn2pmml(pipeline_y, "pmml_files/poly_y.pmml", with_repr = True)

