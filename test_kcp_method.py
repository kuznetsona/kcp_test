import numpy as np
import logging
from typing import List
from kcp_test.core import permutation_test, print_summary
from kcp_test.plotting import plot_time_series, plot_min_variances

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_synthetic_data(num_points: int, list_of_change_points: List[int], correlations: List[float]) -> np.ndarray:
    assert len(list_of_change_points) != len(
        correlations) + 1, "Change points and correlations must have the same length."

    np.random.seed(42)
    time_series = np.zeros((num_points, 2))

    start_point = 0

    for i, change_point in enumerate(list_of_change_points):
        end_point = change_point
        corr = correlations[i]
        time_series[start_point:end_point, 0] = np.random.normal(0, 1, end_point - start_point)
        time_series[start_point:end_point, 1] = corr * time_series[start_point:end_point, 0] + np.random.normal(0, 1,
                                                                                                                end_point - start_point)
        # time_series[start_point:end_point, 2] = corr * time_series[start_point:end_point, 0] + np.random.normal(0, 1, end_point - start_point)
        start_point = end_point

    if start_point < num_points:
        corr = correlations[-1]
        time_series[start_point:, 0] = np.random.normal(0, 1, num_points - start_point)
        time_series[start_point:, 1] = corr * time_series[start_point:, 0] + np.random.normal(0, 1,
                                                                                              num_points - start_point)
        # time_series[start_point:, 2] = corr * time_series[start_point:, 0] + np.random.normal(0, 1,
        #                                                                                       num_points - start_point)

    logging.info(f"Generated time series. Change points {list_of_change_points}. Correlation {correlations}")
    return time_series


if __name__ == "__main__":
    window_size = 25
    max_change_points = 5
    num_permutations = 200

    num_points = 200
    change_points = [50, 100, 150]
    correlations = [0.5, 0.9, 0.3, 0.7]
    time_series = generate_synthetic_data(
        num_points=num_points,
        list_of_change_points=change_points,
        correlations=correlations)

    results = permutation_test(
        time_series,
        max_change_points=max_change_points,
        num_permutations=num_permutations,
        window_size=window_size)

    plot_min_variances(results['all_min_variances'], max_change_points)

    print_summary(time_series, window_size, max_change_points, num_permutations, results)

    plot_time_series(time_series=time_series, window_size=window_size, real_change_points=change_points,
                     predicted_change_points=results['change_points'])
