import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    def to_2d(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:          # single scalar -> one sample, one feature
            return a.reshape(1, 1)
        if a.ndim == 1:          # 1D -> n scalar samples, one feature each
            return a.reshape(-1, 1)
        return a

    train = to_2d(X_train)
    test = to_2d(X_test)
    n_train = train.shape[0]
    n_test = test.shape[0]

    # Pairwise Euclidean distances, shape (n_test, n_train)
    dists = np.sqrt(((test[:, None, :] - train[None, :, :]) ** 2).sum(axis=2))

    # Stable sort keeps lower indices first on ties
    sorted_idx = np.argsort(dists, axis=1, kind="stable")

    k_take = min(k, n_train)
    res = np.full((n_test, k), -1, dtype=int)
    res[:, :k_take] = sorted_idx[:, :k_take]
    return res