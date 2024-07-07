import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed


def running_correlations(data, window_size) -> np.ndarray:
    num_points, num_vars = data.shape
    running_corrs = []

    for i in range(num_points - window_size + 1):
        window = data[i:i + window_size]
        corr_matrix = np.corrcoef(window, rowvar=False)
        lower_triangle_indices = np.tril_indices(num_vars, -1)
        running_corrs.append(corr_matrix[lower_triangle_indices])

    running_corrs = np.array(running_corrs)
    fisher_z = np.arctanh(running_corrs)
    return fisher_z


def compute_kernel_matrix(running_corrs, bandwidth, metric='euclidean') -> np.ndarray:
    pairwise_dists = squareform(pdist(running_corrs, metric))
    kernel_matrix = np.exp(-pairwise_dists ** 2 / (2 * bandwidth ** 2))
    return kernel_matrix


def within_phase_variance(start, end, kernel_matrix) -> float:
    n = end - start
    if n <= 1:
        return float('inf')
    return n - (1 / n) * np.sum(kernel_matrix[start:end, start:end])


def kcp_detection(correlations, max_change_points):
    n = correlations.shape[0]
    D = squareform(pdist(correlations, 'euclidean'))
    median_distance = np.median(D)
    bandwidth = median_distance
    kernel_matrix = compute_kernel_matrix(correlations, bandwidth)

    change_points = [0, n]
    min_variances = [within_phase_variance(0, n, kernel_matrix)]

    for _ in range(1, max_change_points + 1):
        min_variance = float('inf')
        best_new_cp = None
        for i in range(1, n):
            new_cp = sorted(change_points + [i])
            total_variance = sum(within_phase_variance(new_cp[j], new_cp[j + 1], kernel_matrix)
                                 for j in range(len(new_cp) - 1))

            if total_variance < min_variance:
                min_variance = total_variance
                best_new_cp = new_cp

        if best_new_cp:
            change_points = best_new_cp
            min_variances.append(min_variance)

    optimal_change_points = change_points[1:-1]
    min_within_phase_variance = min_variances[-1]

    return optimal_change_points, min_within_phase_variance


def permutation_test(time_series: np.ndarray, max_change_points: int = 2, num_permutations: int = 1000, window_size: int = 15) -> dict:
    def parallel_kcp_detection(permuted_series: np.ndarray) -> tuple[float, float]:
        perm_corrs = running_correlations(permuted_series, window_size)
        _, perm_var = kcp_detection(perm_corrs, max_change_points)
        perm_drop = perm_var - within_phase_variance(0, perm_corrs.shape[0], compute_kernel_matrix(perm_corrs, np.median(squareform(pdist(perm_corrs, 'euclidean')))))
        return perm_var, perm_drop

    original_corrs = running_correlations(time_series, window_size)
    original_optimal_points, original_min_variance = kcp_detection(original_corrs, max_change_points)
    original_drop = original_min_variance - within_phase_variance(0, original_corrs.shape[0], compute_kernel_matrix(original_corrs, np.median(squareform(pdist(original_corrs, 'euclidean')))))

    permuted_results = Parallel(n_jobs=-1)(
        delayed(parallel_kcp_detection)(np.random.permutation(time_series))
        for _ in range(num_permutations)
    )

    permuted_variances, permuted_drops = zip(*permuted_results)

    p_variance_test = np.mean(permuted_variances >= original_min_variance)
    p_drop_test = np.mean(permuted_drops >= original_drop)

    return {
        'change_points': original_optimal_points,
        'p_variance_test': p_variance_test,
        'p_drop_test': p_drop_test
    }


def generate_synthetic_data(num_points=200) -> np.ndarray:
    np.random.seed(42)
    time_series = np.zeros((num_points, 3))

    time_series[:50, 0] = np.random.normal(0, 1, 50)
    time_series[:50, 1] = 0.5 * time_series[:50, 0] + np.random.normal(0, 1, 50)
    time_series[:50, 2] = -0.5 * time_series[:50, 0] + np.random.normal(0, 1, 50)

    time_series[50:100, 0] = np.random.normal(0, 1, 50)
    time_series[50:100, 1] = 0.9 * time_series[50:100, 0] + np.random.normal(0, 1, 50)
    time_series[50:100, 2] = -0.9 * time_series[50:100, 0] + np.random.normal(0, 1, 50)

    time_series[100:150, 0] = np.random.normal(0, 1, 50)
    time_series[100:150, 1] = 0.3 * time_series[100:150, 0] + np.random.normal(0, 1, 50)
    time_series[100:150, 2] = -0.3 * time_series[100:150, 0] + np.random.normal(0, 1, 50)

    time_series[150:, 0] = np.random.normal(0, 1, 50)
    time_series[150:, 1] = 0.7 * time_series[150:, 0] + np.random.normal(0, 1, 50)
    time_series[150:, 2] = -0.7 * time_series[150:, 0] + np.random.normal(0, 1, 50)
    print("Change points: [50, 100, 150]")
    return time_series


if __name__ == "__main__":
    window_size = 15
    time_series = generate_synthetic_data()

    results = permutation_test(time_series,
                               max_change_points=3,
                               num_permutations=1000,
                               window_size=window_size)

    print("Results from Permutation Test:")
    print(results)
