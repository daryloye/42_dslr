#!/usr/bin/env python3
"""
describe.py
"""

import sys
import pandas as pd
import numpy as np


def describe(filepaths):
    """Describe function"""
    # TODO In this part, Professor McGonagall asks you to produce a program called describe.py.
    # TODO This program will take a dataset as a parameter. All it has to do is display information
    # TODO for all numerical features like in the example:
    pass


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        sys.exit(1)
    describe(sys.argv)


if __name__ == "__main__":
    main()
