#!/usr/bin/env python3
"""
describe.py
"""

import sys
import pandas as pd
import numpy as np


def ft_sort(values):
    """Quicksort algorithm"""
    if len(values) <= 1:
        return values
    pivot = values[len(values) // 2]
    left = [x for x in values if x < pivot]
    middle = [x for x in values if x == pivot]
    right = [x for x in values if x > pivot]
    return ft_sort(left) + middle + ft_sort(right)


def ft_percentile(sorted_list, percentile_value):
    position = (len(sorted_list) - 1) * percentile_value
    lower_index = int(position)
    upper_index = lower_index + 1 if lower_index < len(sorted_list) - 1 else lower_index
    fraction = position - lower_index
    return sorted_list[lower_index] + fraction * (sorted_list[upper_index] - sorted_list[lower_index])


def bonus_stats(clean_values, mean, sorted_values):
    """
    - Variance: Average squared deviation from mean (spread of data)
    - Range: Difference between max and min values (data span)
    - IQR: Interquartile Range (Q3 - Q1), middle 50% spread
    - Mode: Most frequently occurring value in the dataset
    - Skewness: Measure of asymmetry (-ve = left skew, +ve = right skew)
    """
    count = len(clean_values)
    
    variance = sum((v - mean) ** 2 for v in clean_values) / count
    
    data_range = sorted_values[-1] - sorted_values[0]
    
    q1 = ft_percentile(sorted_values, 0.25)
    q3 = ft_percentile(sorted_values, 0.75)
    iqr = q3 - q1
    
    freq_dict = {}
    for v in clean_values:
        freq_dict[v] = freq_dict.get(v, 0) + 1
    mode = max(freq_dict, key=freq_dict.get)
    
    std = variance ** 0.5
    skewness = sum((v - mean) ** 3 for v in clean_values) / (count * std ** 3) if std > 0 else 0
    return {
        'Variance': variance,
        'Range': data_range,
        'IQR': iqr,
        'Mode': mode,
        'Skewness': skewness,
    }


def describe(filepath):
    """Initial 8 stats mentioned in pdf"""
    df = pd.read_csv(filepath)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for column_name in numeric_cols:
        clean_values = [value for value in df[column_name] if pd.notna(value)]
        if not clean_values:
            continue
        
        count = 0
        total_sum = 0
        for value in clean_values:
            count += 1
            total_sum += value
        mean = total_sum / count
        
        variance_sum = 0
        for value in clean_values:
            variance_sum += (value - mean) ** 2
        standard_deviation = (variance_sum / count) ** 0.5
        
        sorted_values = ft_sort(clean_values)
        
        column_stats = {
            'Count': float(count),
            'Mean': mean,
            'Std': standard_deviation,
            'Min': sorted_values[0],
            '25%': ft_percentile(sorted_values, 0.25),
            '50%': ft_percentile(sorted_values, 0.50),
            '75%': ft_percentile(sorted_values, 0.75),
            'Max': sorted_values[-1]
        }
        
        bonus = bonus_stats(clean_values, mean, sorted_values)
        column_stats.update(bonus)
        
        stats[column_name] = column_stats
    
    result = pd.DataFrame(stats)
    result = result.reindex(['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max',
                             'Variance', 'Range', 'IQR', 'Mode', 'Skewness'])
    return result


def main():
    """Numerical feature must be transposed to better view the data"""
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        sys.exit(1)
    try:
        result = describe(sys.argv[1])
        result = result.T
        pd.options.display.float_format = '{:.4f}'.format
        pd.options.display.max_columns = None
        pd.options.display.width = None
        print(result.to_string())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
