
# coding: utf-8

# In[1]:


import numpy as np
from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

from hyperopt import Trials, STATUS_OK, rand
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform

from utils import load_data, visualize_history
import json


# In[2]:


def data():
    with open("data/cleaned_idx_dict.json") as f:
        train_idx_dict = json.load(f)
    X_train_1, y_train_1, _ = load_data("data/hallway_train_dec_27.csv", train_idx_dict)
    X_train_2, y_train_2, _ = load_data("data/hallway_train.csv", train_idx_dict)
    X_train = np.vstack((X_train_1, X_train_2))
    y_train = np.vstack((y_train_1, y_train_2))
    X_test, y_test, _ = load_data("data/hallway_test_feb_25.csv", train_idx_dict)

    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    
    return X_train, y_train, X_test, y_test


# In[3]:


def create_model():
    # Hyperparameters
    lr = {{loguniform(-4, -2)}}
    batch_size = {{choice([32, 64, 128, 256])}}
    n_epochs = 500
    
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    model = models.Sequential()

    model.add(layers.Dense({{choice([256, 512, 1024])}}, activation='relu', input_shape=(n_features,)))
    model.add(layers.Dense({{choice([256, 512, 1024])}}, activation='relu'))
    model.add(layers.Dense(2))

    model.compile(optimizer=optimizers.RMSprop(lr=lr),
                  loss='mse',
                  metrics=['mae'])
    
    callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=100, verbose=2, restore_best_weights=True)]

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        callbacks=callbacks, 
                        validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    mae = score[1]
    return {'loss': mae, 'status': STATUS_OK, 'model': model}


# In[4]:


trials = Trials()
best_run, best_model = optim.minimize(model=create_model,
                          data=data,
                          algo=rand.suggest,
                          max_evals=50,
                          trials=trials)


# In[14]:


print("Best loss: {}".format(trials.best_trial["result"]["loss"]))
print("Best hyperparams: ", best_run)


# In[13]:


best_model.save("models/nn.h5")

