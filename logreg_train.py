#!/usr/bin/env python3
"""
logreg_train.py
"""

import sys
import pandas as pd
import numpy as np
import json

feature_names = ['Astronomy', 'Herbology']
hogwarts_house = 'Hogwarts House'
iteration_limit = 10000
learning_rate = 0.1

output_file = "weights.json"

def softmax(z):
    # numerical stability
    max_per_row = np.max(z, axis=1, keepdims=True)
    z -= max_per_row

    exp_z = np.exp(z)
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return y_hat


def logres_train(houses, features, y_actual):
    """Logistic regression train function"""

    m = features.shape[0]   # number of samples
    n = features.shape[1]   # number of features
    k = houses.shape[0]     # number of classes (houses)

    ones_column = np.ones((m, 1))
    x = np.concatenate((ones_column, features), axis = 1)   # (m x n+1)

    theta = np.zeros((n+1, k))                  # (n+1 x k)
    
    for _ in range(iteration_limit):
        z = x @ theta                           # (m x n+1) x (n+1 x k).T = (m x k)
        y_hat = softmax(z)                      # (m x k)
        grad = (x.T @ (y_hat - y_actual)) / m   # (m x n+1).T x (m x k) = (n+1, k)
        theta -= learning_rate * grad

    theta_df = pd.DataFrame(theta, columns=houses, index=(['bias'] + feature_names))
    print(theta_df)
    return theta_df


def save_weights(theta_df):
    theta_json = theta_df.to_json()
    with open(output_file, "w") as file:
        file.write(theta_json)


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)
    try:
        df = pd.read_csv(sys.argv[1], index_col=0)
        df = df.dropna(subset=feature_names)

        houses = df[hogwarts_house].unique()
        features = df[feature_names].to_numpy()     # (m x n) array
        y_actual = pd.get_dummies(df[hogwarts_house]).astype(int).to_numpy()  # one-hot encoding
        
        theta_df = logres_train(houses, features, y_actual)
        save_weights(theta_df)
    except:
        print(f"Error: {e}")
        sys.exit(1) 
 

if __name__ == "__main__":
    main()
