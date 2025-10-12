import numpy as np
import pandas as pd
import json


HOGWARTS_HOUSE = 'Hogwarts House'

FEATURE_NAMES = [
    'Best Hand',
    # 'Arithmancy',
    # 'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    # 'Divination',
    'Muggle Studies',
    # 'Ancient Runes',
    # 'History of Magic',
    # 'Transfiguration',
    # 'Potions',
    # 'Care of Magical Creatures',
    # 'Charms',
    'Flying'
    ]


def softmax(z):
    max_per_row = np.max(z, axis=1, keepdims=True)
    z -= max_per_row        # shift all values so the biggest becomes 0 => exponentials are at most exp(0) = 1
    
    exp_z = np.exp(z)
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return y_hat


def get_df(filename):
    df = pd.read_csv(filename, index_col=0)
    df['Best Hand'] = df['Best Hand'].map({'Left': 0, 'Right': 1})
    for feature in FEATURE_NAMES:
        median_value = df[feature].median()
        df[feature] = df[feature].fillna(median_value)
    return df


# normalise all feature values with min-max scaling
def minmax_scale_train(df):
    mins = df.min()
    maxs = df.max()
    scaled_df = (df - mins) / (maxs - mins)

    scaler = {"mins": mins.to_dict(), "maxs": maxs.to_dict()}
    with open("scaler.json", "w") as f:
        json.dump(scaler, f)
    return scaled_df


def minmax_scale_predict(df):
    with open("scaler.json") as f:
        scaler = json.load(f)
    mins = pd.Series(scaler["mins"])
    maxs = pd.Series(scaler["maxs"])
    scaled_df = (df - mins) / (maxs - mins)
    return scaled_df