#!/usr/bin/env python3
"""
logreg_predict.py
"""

import sys
import pandas as pd
import numpy as np
import json


def logreg_predict(filepaths):
    """Logistic regression for prediction function"""
    # TODO The second one must be named logreg_predict.[extension]. It takes dataset_test.csv
    # TODO as a parameter and a file containing the weights trained by the previous program.
    pass 


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.json")
        sys.exit(1)
    logreg_predict(sys.argv) 
 

if __name__ == "__main__":
    main()
