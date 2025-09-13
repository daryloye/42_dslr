#!/usr/bin/env python3
"""
scatter_plot.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


def scatter_plot(filepath):
    """Scatter plot function"""
    df = pd.read_csv(filepath)

    numeric_cols = [col for col in df.columns if col not in
                    ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']]

    correlations = {}
    for f1, f2 in combinations(numeric_cols, 2):
        clean_data = df[[f1, f2]].dropna()
        if len(clean_data) > 1:
            corr = clean_data[f1].corr(clean_data[f2])
            correlations[(f1, f2)] = abs(corr)

    most_similar = max(correlations.items(), key=lambda x: x[1])
    feature1, feature2 = most_similar[0]

    top_pairs = sorted(correlations.items(), key=lambda x: -x[1])[:4]

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'Top 4 Most Similar Feature Pairs\nMost Similar: {feature1} & {feature2} (r={most_similar[1]:.3f})',
                 fontsize=14, fontweight='bold')

    for idx, ((f1, f2), corr) in enumerate(top_pairs):
        ax = plt.subplot(2, 2, idx + 1)

        if (f1, f2) == most_similar[0]:
            ax.set_facecolor('#e6ffe6')

        if 'Hogwarts House' in df.columns and not df['Hogwarts House'].isna().all():
            houses = df['Hogwarts House'].dropna().unique()
            colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Ravenclaw': 'blue', 'Hufflepuff': 'gold'}

            for house in houses:
                house_data = df[df['Hogwarts House'] == house]
                ax.scatter(house_data[f1], house_data[f2],
                          alpha=0.6, s=3, label=house, color=colors.get(house, 'gray'))

            if idx == 0:
                ax.legend(fontsize=6, loc='best')
        else:
            ax.scatter(df[f1], df[f2], alpha=0.5, s=3, color='purple')

        ax.set_xlabel(f1[:15], fontsize=7)
        ax.set_ylabel(f2[:15], fontsize=7)
        ax.set_title(f'r={corr:.3f}', fontsize=8, fontweight='bold' if (f1, f2) == most_similar[0] else 'normal')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)
    scatter_plot(sys.argv[1])


if __name__ == "__main__":
    main()
