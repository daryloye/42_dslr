#!/usr/bin/env python3
"""
logreg_predict.py
"""

import sys
import pandas as pd
import numpy as np
import json
from logreg_utils import *


output_file = 'houses.csv'


def logreg_predict(test_df, theta_df):
    """Logistic regression for prediction function"""
    
    test_df[FEATURE_NAMES] = minmax_scale_predict(test_df[FEATURE_NAMES])
    x = test_df[FEATURE_NAMES].to_numpy()   # (m x n) array
    x = np.c_[np.ones(x.shape[0]), x]       # add bias column -> (m x n+1)
    
    theta = theta_df.to_numpy()             # (n+1 x k) array
    
    z = x @ theta
    y_hat = softmax(z)
    predicted_house_index = np.argmax(y_hat, axis=1)
    houses = theta_df.columns
    predicted_house = houses[predicted_house_index]
    
    test_df[HOGWARTS_HOUSE] = predicted_house
    y_predict_df = test_df.iloc[:, :1]
    print(y_predict_df)
    return y_predict_df


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.json")
        sys.exit(1)
    try:
        test_df = get_df(sys.argv[1])
        theta_df = pd.read_json(sys.argv[2])
        y_predict_df = logreg_predict(test_df, theta_df)
        y_predict_df.to_csv(output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
 

if __name__ == "__main__":
    main()
