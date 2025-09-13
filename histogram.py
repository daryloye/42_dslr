#!/usr/bin/env python3
"""
histogram.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def histogram(filepath):
    """Histogram function with homogeneity"""
    df = pd.read_csv(filepath)

    if 'Hogwarts House' not in df.columns or df['Hogwarts House'].isna().all():
        print("Error: No house labels found. Use training dataset.")
        sys.exit(1)

    courses = [col for col in df.columns if col not in ['Index', 'Hogwarts House', 'First Name',
                                                         'Last Name', 'Birthday', 'Best Hand']]

    houses = df['Hogwarts House'].dropna().unique()
    colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Ravenclaw': 'blue', 'Hufflepuff': 'yellow'}

    homogeneity_scores = {}
    for course in courses:
        house_scores = []
        for house in houses:
            house_data = df[df['Hogwarts House'] == house][course].dropna()
            if len(house_data) > 0:
                house_scores.append(house_data.values)

        if len(house_scores) >= 2:
            f_stat, p_value = stats.f_oneway(*house_scores)
            homogeneity_scores[course] = p_value

    most_homogeneous = max(homogeneity_scores.items(), key=lambda x: x[1])

    fig = plt.figure(figsize=(16, 10))

    fig.suptitle(f'Hogwarts Course Score Distributions by House\n\n' +
                 f'ANSWER: {most_homogeneous[0]} has the most homogeneous distribution\n' +
                 f'(p-value = {most_homogeneous[1]:.4f}, no significant difference between houses)',
                 fontsize=14, fontweight='bold', y=0.98)

    n_cols = 4
    n_rows = (len(courses) + n_cols - 1) // n_cols

    for idx, course in enumerate(courses):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        all_course_data = df[course].dropna()
        if len(all_course_data) == 0:
            continue

        bins = np.linspace(all_course_data.min(), all_course_data.max(), 20)

        for house in houses[:4]:
            if pd.isna(house):
                continue

            house_data = df[df['Hogwarts House'] == house][course].dropna()
            ax.hist(house_data, bins=bins, alpha=0.5, label=house,
                   color=colors.get(house, 'gray'), edgecolor='black', linewidth=0.3)

        if course == most_homogeneous[0]:
            ax.set_facecolor('#e6ffe6')
            title_color = 'darkgreen'
            title_weight = 'bold'
            title_text = f'{course}\n MOST HOMOGENEOUS'
        else:
            title_color = 'black'
            title_weight = 'normal'
            p_val = homogeneity_scores.get(course, 0)
            title_text = f'{course}\n(p={p_val:.3f})'

        ax.set_title(title_text, fontsize=9, color=title_color, weight=title_weight)
        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python histogram.py dataset_train.csv")
        sys.exit(1)
    histogram(sys.argv[1]) 
 

if __name__ == "__main__":
    main()
