#!/usr/bin/env python3
"""
scatter_plot.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(filepaths):
    """Scatter plot function"""
    # TODO What are the two features that are similar?
    pass


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)
    scatter_plot(sys.argv)
 

if __name__ == "__main__":
    main()
