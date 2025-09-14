#!/usr/bin/env python3
"""
logreg_train.py
"""

import sys
import pandas as pd
import numpy as np
import json

FEATURE_NAMES = ['Astronomy', 'Herbology']
HOGWARTS_HOUSE = 'Hogwarts House'

def softmax(z):
    max_per_row = np.max(z, axis=1, keepdims=True)
    z -= max_per_row        # numerical stability
    
    exp_z = np.exp(z)
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return y_hat


def logres_train(df, mode="batch", batch_size=32,
                learning_rate=0.1, iteration_limit=10000,
                output_file="weights.json"):
    """Logistic regression train function"""

    houses = df[HOGWARTS_HOUSE].unique()
    features = df[FEATURE_NAMES].to_numpy()                                 # (m x n) array
    y_actual = pd.get_dummies(df[HOGWARTS_HOUSE]).astype(int).to_numpy()    # one-hot encoding

    m = features.shape[0]   # number of samples
    n = features.shape[1]   # number of features
    k = houses.shape[0]     # number of classes (houses)

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
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)
    try:
        df = pd.read_csv(sys.argv[1], index_col=0)
        df = df.dropna(subset=FEATURE_NAMES)

        logres_train(df, mode="batch", learning_rate=0.1, iteration_limit=10000, output_file="weights.json")
        logres_train(df, mode="sgd", learning_rate=0.01, iteration_limit=200000, output_file="sgd_weights.json")
        logres_train(df, mode="mini-batch", batch_size=32, learning_rate=0.1, iteration_limit=10000, output_file="mini_batch_weights.json")
    except:
        print(f"Error: {e}")
        sys.exit(1) 
 

if __name__ == "__main__":
    main()
