import numpy as np
from kcp_test.core import KCPDetector
from kcp_test.plotting import plot_time_series, plot_min_variances

def generate_correlated_series(n, corr):
    mean = [0, 0, 0]
    cov = [[1, corr, corr],
           [corr, 1, corr],
           [corr, corr, 1]]
    return np.random.multivariate_normal(mean, cov, size=n)


def generate_synthetic_data(num_points: int = 200) -> np.ndarray:
    np.random.seed(42)
    time_series = np.zeros((num_points, 3))

    corr_values = [0.1, 0.9, 0.3, 0.7]
    segment_length = num_points // len(corr_values)

    for i, corr in enumerate(corr_values):
        start = i * segment_length
        end = start + segment_length
        time_series[start:end] = generate_correlated_series(segment_length, corr)

    print("Change points:", [segment_length * i for i in range(1, len(corr_values))])
    return time_series


if __name__ == "__main__":
    window_size = 50
    max_change_points = 7
    num_permutations = 1000

    time_series = generate_synthetic_data(200)

    detector = KCPDetector(max_change_points=max_change_points,
                           window_size=window_size,
                           num_permutations=num_permutations)
    detector.fit(time_series)
    results = detector.permutation_test(time_series)

    print("Change points detected:", results['change_points'])
    print("P-value (variance test):", results['p_variance_test'])
    print("P-value (drop test):", results['p_drop_test'])

    plot_time_series(time_series=time_series,
                     window_size=window_size,
                     real_change_points=[50, 100, 150],
                     predicted_change_points=results['change_points'])

    # plot_time_series(time_series=time_series, window_size=window_size, real_change_points=[50, 100, 150],
    #                  predicted_change_points=[50, 100, 150])
