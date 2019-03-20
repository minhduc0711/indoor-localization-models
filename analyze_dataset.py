
# coding: utf-8

# In[7]:


import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


def load_data(path):
    df_list = []

    with open(path) as f:
        str_list = f.read().split("\n\n")
    for s in str_list:
        df_list.append(pd.read_csv(io.StringIO(s), header=None))
    return df_list


# In[16]:


df_list = load_data("data/hallway_train.csv")
df_list.extend(load_data("data/hallway_train_dec_27.csv"))


# In[25]:


df = pd.concat(df_list, ignore_index=True)
signal_arr = np.asarray(df[3])


# In[76]:


names = ["[0 11)", "[11 22)", "[22 33)", "[33 44)", "[44 55)", "[55 66)", "[66 77)", "[77 88)", "[88 99]"]

fig, ax = plt.subplots()
bins = np.histogram(signal_arr, bins=[0, 11, 22, 33, 44, 55, 66, 77, 88, 99])[0]
ax.bar(np.arange(9), bins)
ax.set_xticks(np.arange(9))
ax.set_xticklabels(names, rotation=45, rotation_mode="anchor", ha="right")
ax.set_xlabel("Signal level")
ax.set_ylabel("Number of WAPs detected")

plt.show()

