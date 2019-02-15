import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def create_idx_dict(df_list):
    cnt = 0
    idx_dict = {}
    for df in df_list:
        for _, row in df.iterrows():
            if idx_dict.get(row[2]) is None:
                idx_dict[row[2]] = cnt
                cnt += 1
    return idx_dict


def to_feature_vector(df, idx_dict):
    n_features = len(idx_dict) + 1

    xi = np.zeros((n_features,))
    for _, row in df.iterrows():
        idx = idx_dict.get(row[2])
        if idx is not None:
            xi[idx] = row[3] / 99.
    xi[-1] = df.iloc[0, 4] / 360.
    return xi.reshape(1, -1)


def load_data(path, idx_dict=None):
    df_list = []

    with open(path) as f:
        str_list = f.read().split("\n\n")
    for s in str_list:
        df_list.append(pd.read_csv(io.StringIO(s), header=None))

    if idx_dict is None:
        idx_dict = create_idx_dict(df_list)

    n_samples = len(df_list)
    n_features = len(idx_dict) + 1
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, 2))

    for (i, df) in enumerate(df_list):
        X[i, :] = to_feature_vector(df, idx_dict)
        # Swap positions of x and y co-ords 
        y[i, 0] = df.iloc[0, 1]
        y[i, 1] = df.iloc[0, 0]
    return X, y, idx_dict


def visualize_history(history):
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()


def remove_noisy_ssid(fnames):
    idx_dicts = []
    for fname in fnames:
        _, _, idx_dict = load_data(fname)
        idx_dicts.append(idx_dict)

    common_keys = set.intersection(*map(set, idx_dicts))
    cleaned_idx_dict = {}
    idx = 0
    for ssid in list(common_keys):
        cleaned_idx_dict[ssid] = idx
        idx += 1
    
    with open("cleaned_idx_dict.json", "w") as f:
        json.dump(cleaned_idx_dict, f)


# if __name__ == "__main__":
    # remove_noisy_ssid(["data/hallway_train.csv", "data/hallway_train_dec_27.csv"])