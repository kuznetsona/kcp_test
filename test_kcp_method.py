import numpy as np
from kcp_test.core import permutation_test, print_summary
from kcp_test.plotting import plot_time_series, plot_min_variances


# TODO Обобщить генерацию данных; например, до сигнатуры (list_of_change_points: List[int])
def generate_synthetic_data(num_points: int = 200) -> np.ndarray:
    np.random.seed(42)
    time_series = np.zeros((num_points, 3))

    time_series[:50, 0] = np.random.normal(0, 1, 50)
    time_series[:50, 1] = 0.5 * time_series[:50, 0] + np.random.normal(0, 1, 50)
    time_series[:50, 2] = 0.5 * time_series[:50, 0] + np.random.normal(0, 1, 50)

    time_series[50:100, 0] = np.random.normal(0, 1, 50)
    time_series[50:100, 1] = 0.9 * time_series[50:100, 0] + np.random.normal(0, 1, 50)
    time_series[50:100, 2] = 0.9 * time_series[50:100, 0] + np.random.normal(0, 1, 50)

    time_series[100:150, 0] = np.random.normal(0, 1, 50)
    time_series[100:150, 1] = 0.3 * time_series[100:150, 0] + np.random.normal(0, 1, 50)
    time_series[100:150, 2] = 0.3 * time_series[100:150, 0] + np.random.normal(0, 1, 50)

    time_series[150:, 0] = np.random.normal(0, 1, 50)
    time_series[150:, 1] = 0.7 * time_series[150:, 0] + np.random.normal(0, 1, 50)
    time_series[150:, 2] = 0.7 * time_series[150:, 0] + np.random.normal(0, 1, 50)
    print("Change points: [50, 100, 150]")
    return time_series


if __name__ == "__main__":
    window_size = 25
    max_change_points = 7
    num_permutations = 200

    time_series = generate_synthetic_data()

    results = permutation_test(
        time_series,
        max_change_points=max_change_points,
        num_permutations=num_permutations,
        window_size=window_size)

    plot_min_variances(results['all_min_variances'], max_change_points)

    print_summary(time_series, window_size, max_change_points, num_permutations, results)

    plot_time_series(time_series=time_series, window_size=window_size, real_change_points=[50, 100, 150],
                     predicted_change_points=results['change_points'])
