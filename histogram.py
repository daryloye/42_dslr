#!/usr/bin/env python3
"""
histogram.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def histogram(filepath):
    """Histogram function"""
    # TODO Which Hogwarts course has a homogeneous score distribution between all four houses?
    pass


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python histogram.py dataset_train.csv")
        sys.exit(1)
    histogram(sys.argv) 
 

if __name__ == "__main__":
    main()
