#!/usr/bin/env python3
"""
logreg_train.py
"""

import sys
import pandas as pd
import numpy as np
import json


def logres_train(filepaths):
    """Logistic regression train function"""
    # TODO The first one will train your models, and it’s called logreg_train.py.
    # TODO It takes dataset_train.csv as a parameter. For the mandatory part, you must
    # TODO use the technique of gradient descent to minimize the error. The program generates
    # TODO a file containing the weights that will be used for the prediction.
    pass 


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)
    logreg_train(sys.argv) 
 

if __name__ == "__main__":
    main()
