#!/usr/bin/env python3
"""
logreg_predict.py
"""

import sys
import pandas as pd
import numpy as np
import json

hogwarts_house = 'Hogwarts House'
output_file = 'houses.csv'

def softmax(z):
    # numerical stability
    max_per_row = np.max(z, axis=1, keepdims=True)
    z -= max_per_row

    exp_z = np.exp(z)
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return y_hat


def logreg_predict(test_df, theta_df):
    """Logistic regression for prediction function"""

    feature_names = theta_df.index[1:]
    
    x = test_df[feature_names].to_numpy()   # (m x n)
    x = np.c_[np.ones(x.shape[0]), x]       # add bias column -> (m x n+1)
    
    theta = theta_df.to_numpy()             # (n+1 x k)
    
    z = x @ theta
    y_hat = softmax(z)
    predicted_index = np.argmax(y_hat, axis=1)
    predicted_class = theta_df.columns[predicted_index]
    
    test_df[hogwarts_house] = predicted_class
    y_predict_df = test_df.iloc[:, :1]
    print(y_predict_df)
    return y_predict_df


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.json")
        sys.exit(1)
    try:
        test_df = pd.read_csv(sys.argv[1], index_col=0)
        theta_df = pd.read_json(sys.argv[2])
        y_predict_df = logreg_predict(test_df, theta_df)
        y_predict_df.to_csv(output_file)
    except:
        print(f"Error: {e}")
        sys.exit(1)
 

if __name__ == "__main__":
    main()
