
# coding: utf-8

# In[20]:


import json
from utils import load_data

import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model


# In[14]:


with open("data/cleaned_idx_dict.json") as f:
    train_idx_dict = json.load(f)
X_test, y_test, _ = load_data("./data/hallway_test_feb_25.csv", train_idx_dict)

model = load_model("models/neural_net.h5")
y_pred = model.predict(X_test)


# In[30]:


img = plt.imread("images/hallway.png")


# In[54]:


x_plot = y_test[:, 0] * (img.shape[1] / 20)
y_plot = y_test[:, 1] * (img.shape[0] / 40)
errors = np.sum(np.abs(y_test - y_pred), axis=1)


f, ax = plt.subplots()
plt.imshow(img, origin="upper")

points = ax.scatter(x_plot, y_plot, c=errors, cmap='coolwarm', s=15)
f.colorbar(points)

ax.get_xaxis().set_ticklabels([])
ax.get_yaxis().set_ticklabels([])
plt.show()
