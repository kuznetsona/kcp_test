import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from typing import List, Tuple, Dict


def running_correlations(data: np.ndarray, window_size: int) -> np.ndarray:
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


def compute_kernel_matrix(running_corrs: np.ndarray, bandwidth: float) -> np.ndarray:
    pairwise_dists = squareform(pdist(running_corrs, 'euclidean'))
    kernel_matrix = np.exp(-pairwise_dists ** 2 / (2 * bandwidth ** 2))
    return kernel_matrix


def within_phase_variance(start: int, end: int, kernel_matrix: np.ndarray) -> float:
    n = end - start
    if n <= 1:
        return float('inf')
    return n - (1 / n) * np.sum(kernel_matrix[start:end, start:end])


def locate_cp(H: np.ndarray, ncp: int, wsize: int) -> np.ndarray:
    wstep = 1
    d = ncp + 1
    cps = [H.shape[1] - 1]
    cp = cps[0]

    while d > 0:
        cp = H[d, cp]
        cps.insert(0, cp)
        d -= 1

    cps = np.array(cps) + 1

    if wsize % 2 > 0:
        cps = np.ceil(wsize / 2) + (cps - 1) * wstep
    else:
        cps = np.ceil((wsize / 2 + 0.5) + (cps - 1) * wstep)

    cps = cps.astype(int)
    return cps[1:-1]


def get_scatter_matrix(II_: np.ndarray, X_: np.ndarray, H_: np.ndarray):
    X = X_
    M = X.shape[1]
    N = X.shape[0]
    u = np.zeros((N, N))
    K = np.zeros((N, N))
    full_cumsum = np.zeros((N + 1, N + 1))

    II = II_
    H = H_

    for i in range(N):
        for j in range(i, N):
            total = 0.0
            for k in range(M):
                diff = X[i, k] - X[j, k]
                total += diff * diff
            u[i, j] = u[j, i] = total

    medianK = np.median(u)
    median_times_two = 2 * medianK

    if median_times_two != 0.0:
        for i in range(N):
            for j in range(i + 1):
                K[j, i] = -u[j, i] / median_times_two

        full_cumsum[0, N] = 0.0
        for i in range(N):
            top = 0.0
            left_top = 0.0
            for j in range(i + 1):
                left = full_cumsum[j + 1, i]
                full_cumsum[j + 1, i + 1] = top = left + top + np.exp(K[j, i]) - left_top
                left_top = left
            full_cumsum[i + 1, i + 1] = full_cumsum[i, i + 1] + full_cumsum[i, i + 1] + np.exp(K[i, i]) - full_cumsum[
                i, i]

        diag_cumsum = np.diag(full_cumsum)

        for j in range(N):
            full_cumsum_j_j = diag_cumsum[j + 1]
            for i in range(j):
                temp = j - i + 1.0
                K[i, j] = temp - (full_cumsum_j_j + diag_cumsum[i] - 2.0 * full_cumsum[i, j + 1]) / temp

        L = H.shape[0]

        for i in range(N):
            II[0, i] = K[0, i]

        for k in range(1, L):
            for i in range(k, N):
                for j in range(k - 1, i):
                    tmp = II[k - 1, j] + K[j + 1, i]
                    if tmp < II[k, i]:
                        II[k, i] = tmp
                        H[k, i] = j + 1

    return II, H, medianK


def elbow_method(min_variances: List[float]) -> int:
    n_points = len(min_variances)
    all_coord = np.vstack((range(n_points), min_variances)).T
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))

    vec_from_first = all_coord - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel

    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    optimal_idx = np.argmax(dist_to_line)

    return optimal_idx + 1


def kcp_detection(correlations: np.ndarray, max_change_points: int):
    n = correlations.shape[0]
    D = max_change_points + 1
    II_ = np.full((D, n), np.inf)
    H_ = np.zeros((D, n), dtype=int)

    II, H, medianK = get_scatter_matrix(II_, correlations, H_)

    best_change_points = []
    best_min_variance = []

    if medianK == 0:
        return best_change_points, best_min_variance

    for k in range(1, max_change_points + 1):
        change_points = [0, n]
        min_variances = [within_phase_variance(0, n, compute_kernel_matrix(correlations, medianK))]

        for _ in range(1, k + 1):
            min_variance = float('inf')
            best_new_cp = None
            for i in range(1, n):
                new_cp = sorted(change_points + [i])
                total_variance = sum(
                    within_phase_variance(new_cp[j], new_cp[j + 1], compute_kernel_matrix(correlations, medianK))
                    for j in range(len(new_cp) - 1))

                if total_variance < min_variance:
                    min_variance = total_variance
                    best_new_cp = new_cp

            if best_new_cp:
                change_points = best_new_cp
                min_variances.append(min_variance)

        optimal_change_points = change_points[1:-1]
        min_within_phase_variance = min_variances[-1]

        best_min_variance.append(min_within_phase_variance)
        best_change_points.append(optimal_change_points)

    optimal_k = elbow_method(best_min_variance)
    return best_change_points, best_min_variance, optimal_k


def permutation_test(time_series: np.ndarray, max_change_points: int = 2, num_permutations: int = 1000,
                     window_size: int = 15) -> Dict[str, float]:
    def parallel_kcp_detection(permuted_series: np.ndarray) -> Tuple[float, float]:
        perm_corrs = running_correlations(permuted_series, window_size)
        perm_change_points, perm_variances, perm_optimal_k = kcp_detection(perm_corrs, max_change_points)
        perm_var = perm_variances[perm_optimal_k - 1]

        perm_kernel_matrix = compute_kernel_matrix(perm_corrs, np.median(squareform(pdist(perm_corrs))))
        perm_drop = perm_var - within_phase_variance(0, perm_corrs.shape[0], perm_kernel_matrix)
        return perm_var, perm_drop

    original_corrs = running_correlations(time_series, window_size)
    original_change_points, original_min_variances, original_optimal_k = kcp_detection(original_corrs,
                                                                                       max_change_points)

    original_var = original_min_variances[original_optimal_k - 1]
    original_kernel_matrix = compute_kernel_matrix(original_corrs, np.median(squareform(pdist(original_corrs))))
    original_drop = original_var - within_phase_variance(0, original_corrs.shape[0], original_kernel_matrix)

    permuted_results = Parallel(n_jobs=-1)(
        delayed(parallel_kcp_detection)(np.random.permutation(time_series))
        for _ in range(num_permutations)
    )

    permuted_variances, permuted_drops = zip(*permuted_results)

    p_variance_test = np.mean(permuted_variances >= original_var)
    p_drop_test = np.mean(permuted_drops >= original_drop)

    return {
        'change_points': original_change_points[original_optimal_k - 1] if original_change_points else [],
        'p_variance_test': p_variance_test,
        'p_drop_test': p_drop_test,
        'all_change_points': original_change_points,
        'all_min_variances': original_min_variances
    }


def print_summary(time_series: np.ndarray, window_size: int, max_change_points: int, num_permutations: int,
                  results: Dict[str, float]) -> None:
    print("SETTINGS:")
    print(f"    Number of running Correlations monitored: {time_series.shape[1]}")
    print(f"    Selected window size: {window_size}")
    print(f"    Selected maximum number of change points: {max_change_points}")
    print(f"        Number of permuted data sets: {num_permutations}\n")

    print("OUTPUT:")
    print(f"    Number of change points detected based on grid search: {len(results['change_points'])}")
    print(f"    Change point location(s): {', '.join(map(str, results['change_points']))}")
    print()
    print(f"    Number of change points detected based on scree test: {len(results['change_points'])}")

    print(f"    Significance level of each subtest: 0.025")
    print(f"         P-value of the variance test: {results['p_variance_test']:.4f}")
    print(f"         P-value of the variance drop test: {results['p_drop_test']:.4f}")
