#!/usr/bin/env python3
"""
pair_plot.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def pair_plot(filepaths):
    """Pair plot function"""
    # TODO From this visualization, which features are you going to use for your logistic regression?
    pass


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset_train.csv")
        sys.exit(1)
    pair_plot(sys.argv) 
 

if __name__ == "__main__":
    main()
