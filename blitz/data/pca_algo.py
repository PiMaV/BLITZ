from typing import Tuple, Optional
import numpy as np

def randomized_svd_low_memory(
    matrix: np.ndarray,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Randomized SVD without explicitly forming the centered matrix.

    A is (n_samples, n_features).
    Returns (U, s, Vh, mean).

    Memory efficient approach:
    Instead of C = A - mean (which duplicates memory),
    we compute Y = C @ Omega algebraically as:
    Y = (A @ Omega) - (mean @ Omega)

    Args:
        matrix: Input matrix (n_samples, n_features)
        n_components: Number of components to keep
        n_oversamples: Additional dimensions for stability
        n_iter: Power iterations (improves accuracy)
        random_state: Seed for reproducibility

    Returns:
        U: (n_samples, n_components)
        s: (n_components,)
        Vh: (n_components, n_features)
        mean: (n_features,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = matrix.shape
    k = n_components + n_oversamples

    # 1. Compute Mean (O(MN) but can be done sequentially or just standard numpy)
    # We rely on numpy's optimization here.
    mean = np.mean(matrix, axis=0)

    # 2. Generate Random Matrix Omega (N, k)
    # Standard normal distribution
    # Critical: Omega must be float (typically float32 or float64) to maintain
    # Gaussian properties. Do not cast to matrix.dtype if matrix is integer.
    dtype = matrix.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32

    omega = np.random.normal(size=(n_features, k)).astype(dtype)

    # 3. Compute Y = (A - mean) @ Omega = A @ Omega - mean @ Omega
    # A @ Omega is (M, N) * (N, k) -> (M, k)
    Y = matrix @ omega

    # mean @ Omega is (N,) * (N, k) -> (k,)
    # We broadcast this subtraction
    mean_correction = mean @ omega
    Y -= mean_correction

    # 4. Power Iterations (optional, for better singular vector decay)
    # Y = (A - mean) (A - mean).T Y
    # Again, avoiding explicit (A - mean).
    # (A - mean).T Y = A.T Y - mean.T Y (outer product sum?)
    # mean.T Y is (N, 1) * (1, M) * (M, k)? No.
    # mean (N,). Y (M, k).
    # (A - mean).T @ Y = A.T @ Y - mean.T @ Y
    # A.T @ Y is (N, M) * (M, k) -> (N, k)
    # mean is (N,). Y is (M, k). Sum of Y cols is (k,).
    # mean.reshape(N, 1) @ Y.sum(axis=0).reshape(1, k) -> (N, k)

    for _ in range(n_iter):
        # Y_new = (A - mean).T @ Y
        #       = A.T @ Y - mean.reshape(-1, 1) @ np.sum(Y, axis=0).reshape(1, -1)
        Y_sum = np.sum(Y, axis=0)
        Y = matrix.T @ Y
        Y -= np.outer(mean, Y_sum)

        # QR for stability
        Y, _ = np.linalg.qr(Y, mode='reduced')

        # Y_new = (A - mean) @ Y
        #       = A @ Y - mean @ Y
        # A @ Y is (M, N) * (N, k) -> (M, k)
        # mean @ Y is (N,) * (N, k) -> (k,)
        mean_corr = mean @ Y
        Y = matrix @ Y
        Y -= mean_corr

        # QR again
        Y, _ = np.linalg.qr(Y, mode='reduced')

    # 5. Form Q (Orthonormal basis for range of centered A)
    # Y is already Q after the loop if n_iter > 0.
    # If n_iter == 0, we do one QR now.
    Q, _ = np.linalg.qr(Y, mode='reduced')

    # 6. Project data into subspace: B = Q.T @ (A - mean)
    # B = Q.T @ A - Q.T @ mean
    # Q is (M, k). B is (k, N).
    # Q.T @ A is (k, M) * (M, N) -> (k, N)
    B = Q.T @ matrix

    # Q.T @ mean? No, mean is (N,).
    # We want Q.T @ (A - mean_row_vector).
    # This matches.
    # Q.T @ A  (k, N)
    # Q.T @ (ones(M,1) * mean) = (Q.T @ ones) * mean
    # Q.T sum over columns.
    q_sum = np.sum(Q, axis=0) # (k,)
    B -= np.outer(q_sum, mean)

    # 7. Compute SVD of small matrix B
    # B is (k, N). k is small (~10-50). N is large (~10000).
    # SVD(B) -> U_tilde (k, k), s (k,), Vh (k, N)
    U_tilde, s, Vh = np.linalg.svd(B, full_matrices=False)

    # 8. Recover U = Q @ U_tilde
    U = Q @ U_tilde

    # 9. Truncate to n_components
    return (
        U[:, :n_components],
        s[:n_components],
        Vh[:n_components, :],
        mean
    )

def svd_exact(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper for exact SVD. Creates centered copy (Memory intensive!).
    """
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    U, s, Vh = np.linalg.svd(centered, full_matrices=False)
    return U, s, Vh, mean
