from typing import List
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from kcp_test.core import KCPDetector

def plot_time_series(time_series: np.ndarray,
                     window_size: int,
                     real_change_points: List[int],
                     predicted_change_points: List[int]) -> None:
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    for i in range(time_series.shape[1]):
        plt.plot(time_series[:, i], label=f'Series {i + 1}')

    for cp in real_change_points:
        plt.axvline(x=cp, color='red', linestyle='--', label='Real Change Point')

    for pcp in predicted_change_points:
        plt.axvline(x=pcp, color='green', linestyle='--', label='Predicted Change Point')

    plt.title('Time Series with Real and Predicted Change Points')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)

    detector = KCPDetector(max_change_points=2, window_size=window_size, num_permutations=100)

    running_corrs = detector.running_correlations(time_series)

    num_series = time_series.shape[1]
    pair_indices = [(i, j) for i in range(num_series) for j in range(i + 1, num_series)]

    for idx, (i, j) in enumerate(pair_indices):
        plt.plot(running_corrs[:, idx], label=f'Correlation {i + 1}-{j + 1}')

    for cp in real_change_points:
        plt.axvline(x=cp, color='red', linestyle='--')

    for pcp in predicted_change_points:
        plt.axvline(x=pcp, color='green', linestyle='--')

    plt.title('Running Correlations with Real and Predicted Change Points')
    plt.xlabel('Time Window')
    plt.ylabel('Correlation')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_min_variances(min_variances: List[float], max_change_points: int) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, max_change_points + 1), min_variances, marker='o', linestyle='-')

    plt.title('Minimized Within-Phase Variance for Each Number of Change Points')
    plt.xlabel('Number of Change Points')
    plt.ylabel('Minimized Within-Phase Variance')
    plt.grid(True)
    plt.show()
