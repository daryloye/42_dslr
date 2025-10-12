#!/usr/bin/env python3
"""
logreg_train.py
"""

import sys
import pandas as pd
import numpy as np
import json
from logreg_utils import *


def logres_train(df, mode="batch", batch_size=32,
                learning_rate=0.1, iteration_limit=10000,
                output_file="weights.json"):
    """Logistic regression train function"""

    df[FEATURE_NAMES] = minmax_scale_train(df[FEATURE_NAMES])
    features = df[FEATURE_NAMES].to_numpy()   # (m x n) array
    y_onehot_df = pd.get_dummies(df[HOGWARTS_HOUSE])
    houses = list(y_onehot_df.columns)                
    y_actual = y_onehot_df.astype(int).to_numpy()    # one-hot encoding

    m = features.shape[0]   # number of samples
    n = features.shape[1]   # number of features
    k = len(houses)         # number of classes (houses)

    ones_column = np.ones((m, 1))
    x = np.concatenate((ones_column, features), axis = 1)   # (m x n+1)

    theta = np.zeros((n+1, k))                              # (n+1 x k)
    
    for _ in range(iteration_limit):
        if mode == "batch":
            z = x @ theta                           # (m x n+1) x (n+1 x k) = (m x k)
            y_hat = softmax(z)                      # (m x k)
            grad = (x.T @ (y_hat - y_actual)) / m   # (m x n+1).T x (m x k) = (n+1, k)
            theta -= learning_rate * grad
        
        elif mode == "sgd":
            # Stochastic gradient descent
            # Updates the model's parameters using one random training sample at a time.

            i = np.random.randint(0, m)
            z_i = x[i] @ theta                               # (n+1, ) x (n+1 x k) = (k, )
            y_hat_i = softmax(z_i.reshape(1, -1))[0]         # (1, k) -> (k, )
            grad = np.outer(x[i], (y_hat_i - y_actual[i]))   # (n+1, ) x (k, ) = (n+1, k)
            theta -= learning_rate * grad
            
            # x[i] is 1-D array, cannot do @ multiplication
            # .reshape(1, -1) transforms the 1-D array (k, ) into 2-D (1, k)

        elif mode == "mini-batch":
            # Mini-batch GD
            # Updates the model’s parameters after processing a mini-batch of data 

            i = np.random.choice(m, batch_size, replace=False)
            x_batch = x[i]
            y_batch = y_actual[i]

            z = x_batch @ theta
            y_hat = softmax(z)
            grad = (x_batch.T @ (y_hat - y_batch)) / batch_size
            theta -= learning_rate * grad
        
        else:
            raise ValueError("mode must be 'batch', 'mini-batch', or 'sgd'")

    theta_df = pd.DataFrame(theta, columns=houses, index=(['bias'] + FEATURE_NAMES))
    theta_df.to_json(output_file)
    print("\033[34m" + mode + "\033[0m")  
    print(theta_df)
    print()


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python logreg_train.py dataset_train.csv batch|sgd|mini-batch")
        sys.exit(1)
    try:
        df = get_df(sys.argv[1])
        if sys.argv[2] == "batch":
            logres_train(df, mode="batch", learning_rate=0.5, iteration_limit=3000, output_file="weights.json")
        elif sys.argv[2] == "sgd":
            logres_train(df, mode="sgd", learning_rate=0.1, iteration_limit=100000, output_file="weights.json")
        elif sys.argv[2] == "mini-batch":
            logres_train(df, mode="mini-batch", batch_size=32, learning_rate=0.5, iteration_limit=3000, output_file="weights.json")
        else:
            raise ValueError("argument must be 'batch', 'mini-batch', or 'sgd'")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 
 

if __name__ == "__main__":
    main()
