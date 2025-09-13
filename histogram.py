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
    df = pd.read_csv(filepath)
    
    if 'Hogwarts House' not in df.columns or df['Hogwarts House'].isna().all():
        print("Error: No house labels found. Use training dataset.")
        sys.exit(1)
    
    courses = [col for col in df.columns if col not in ['Index', 'Hogwarts House', 'First Name', 
                                                         'Last Name', 'Birthday', 'Best Hand']]
    
    houses = df['Hogwarts House'].dropna().unique()
    colors = {'Gryffindor': 'red', 'Slytherin': 'green', 
              'Ravenclaw': 'blue', 'Hufflepuff': '#FFD700'}
    
    for course in courses:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{course} - Score Distribution by House', fontsize=16)
        
        axes = axes.flatten()
        
        all_course_data = df[course].dropna()
        bins = np.linspace(all_course_data.min(), all_course_data.max(), 20)
        
        for idx, house in enumerate(houses[:4]):
            if pd.isna(house):
                continue
                
            ax = axes[idx]
            house_data = df[df['Hogwarts House'] == house][course].dropna()
            
            ax.hist(house_data, bins=bins, color=colors.get(house, 'gray'), 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{house}', fontsize=12, color=colors.get(house, 'black'))
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            ax.text(0.02, 0.98, f'n={len(house_data)}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python histogram.py dataset_train.csv")
        sys.exit(1)
    histogram(sys.argv[1]) 
 

if __name__ == "__main__":
    main()
