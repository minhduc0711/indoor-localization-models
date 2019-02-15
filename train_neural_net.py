import numpy as np
from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

import json
from utils import load_data, visualize_history

with open("cleaned_idx_dict.json") as f:
    train_idx_dict = json.load(f)
X_train, y_train, _ = load_data("data/hallway_train.csv", train_idx_dict)
X_test, y_test, _ = load_data("data/hallway_test.csv", train_idx_dict)


print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))

# lr = 2.5e-2
lr = 1e-03
batch_size = 32
n_epochs = 500

n_samples = X_train.shape[0]
n_features = X_train.shape[1]

model = models.Sequential()

model.add(layers.Dense(1024, activation='relu', input_shape=(n_features,)))
model.add(layers.Dense(1024, activation='relu'))
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
visualize_history(history)

model.save('neural_net.h5')
