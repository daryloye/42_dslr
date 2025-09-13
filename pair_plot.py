#!/usr/bin/env python3
"""
pair_plot.py - Pair plot matrix for feature selection
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pair_plot(filepath):
    """Pair plot function"""
    df = pd.read_csv(filepath)

    features = [col for col in df.columns if col not in
                ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']]

    for start_idx in range(0, len(features), 4):
        subset = features[start_idx:start_idx+4]

        if len(subset) < 2:
            continue

        if 'Hogwarts House' in df.columns and not df['Hogwarts House'].isna().all():
            plot_data = df[subset + ['Hogwarts House']].dropna()
            g = sns.pairplot(plot_data, hue='Hogwarts House',
                            palette={'Gryffindor': 'red', 'Slytherin': 'green',
                                    'Ravenclaw': 'blue', 'Hufflepuff': 'gold'},
                            diag_kind='hist', corner=False,
                            plot_kws={'alpha': 0.4, 's': 3})
        else:
            plot_data = df[subset].dropna()
            g = sns.pairplot(plot_data, diag_kind='hist', corner=False,
                            plot_kws={'alpha': 0.4, 's': 3, 'color': 'purple'})

        plt.suptitle(f'Pair Plot - Features {start_idx+1} to {min(start_idx+4, len(features))}\n' +
                     f'{", ".join([f[:15] for f in subset])}\n' +
                     'Features that show clear house separation are best for logistic regression',
                     y=1.01, fontsize=10)
        plt.tight_layout()
        plt.show()


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset_train.csv")
        sys.exit(1)
    pair_plot(sys.argv[1])


if __name__ == "__main__":
    main()
