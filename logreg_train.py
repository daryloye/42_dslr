#!/usr/bin/env python3
"""
logreg_train.py
"""

import sys
import pandas as pd
import numpy as np
import json

iterationLimit = 10000
learningRate = 0.1

outputFile = "weights.json"

def logres_train(filepath):
    """Logistic regression train function"""
    # TODO The first one will train your models, and it’s called logreg_train.py.
    # TODO It takes dataset_train.csv as a parameter. For the mandatory part, you must
    # TODO use the technique of gradient descent to minimize the error. The program generates
    # TODO a file containing the weights that will be used for the prediction.

    x_array = []
    y_array = []

    try:
        with open(filepath) as file:
            for line in file:
                x, y = line.strip().split(",")  # TODO: choose columns for x and y

                try:
                    x_array.append(float(x))
                    y_array.append(float(y))
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print("csv file not found")
        exit(1)

    # normalise data
    n = len(x_array)
    min_x = min(x_array)
    max_x = max(x_array)
    x_norm = [(n - min_x) / (max_x - min_x) for n in x_array]

    # gradient descent loop
    raw_theta0 = 0
    raw_theta1 = 0

    # TODO: use log regression formula
    def getEstimateY(x):
        return raw_theta0 + (raw_theta1 * x)
    
    for _ in range(iterationLimit):
        sum0 = sum(getEstimateY(x_norm[i]) - y[i] for i in range(n))
        sum1 = sum((getEstimateY(x_norm[i]) - y[i]) * x_norm[i] for i in range(n))

        raw_theta0 -= learningRate * 1/n * sum0
        raw_theta1 -= learningRate * 1/2 * sum1
    
    # adjusted theta
    theta1 = raw_theta1 / (max_x - min_x)
    theta0 = raw_theta0 - theta1 * min_x

    print(json.dumps({'theta0': theta0, 'theta1': theta1}))
    # TODO: write to file

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)
    logreg_train(sys.argv[1]) 
 

if __name__ == "__main__":
    main()
