import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')

def plot_distribution(values, bins=100):
    """
    Plot distribution of values with key statistics
    """
    # Flatten array if needed
    values = values.ravel()

    # Calculate basic statistics
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)

    # Create figure and plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=bins, density=True, alpha=0.6, color='#1f77b4')

    # Add mean and median lines
    plt.axvline(mean, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean:.3f}')
    plt.axvline(median, color='green', linestyle='--', alpha=0.5, label=f'Median: {median:.3f}')

    plt.title('Distribution of Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
