import numpy as np
import pandas as pd


# import sklearn.metrics as skm

# Use these four
# kld (Kullback-Leibler Divergence). Created in this file.
# skm.mutual_info_score(data1, data2) #Entropy-based. Using instead of making a function of autoencoder size to score.
# scipy.stats.wasserstein_distance # Earth Mover's Distance, used in WGANs
# compare_time_by_granger_causality (Granger Causality). Created in this file.

# And select one from here or make your own
# skm.pairwise.cosine_similarity
# L2 norm: np.linalg.norm(x,ord=2) # Euclidean
# L1 norm: np.linalg.norm(x,ord=1) # Manhattan


def kld(p: pd.DataFrame, q: pd.DataFrame, bins=35) -> float:
    from scipy.stats import entropy
    """
    Compute the Kullback-Leibler Divergence using scipy's entropy function. KLD(p||q).
    This is an assymetric measure. KLD(p||q) can be understood as the amount of information lost when q is used to approximate p.
    So, a lower result implies q is a better approximation of p.
    If there's time, I'll partition PDFs by sine/cosine pairs and each categorical feature. 
    There are pros and cons to partitioning and this holistic approach.
    Don't worry about that though, just plop in dataframes and expect a float to return.
    """
    # Estimate PDFs
    p_hist = np.histogram(p.to_numpy(), bins, density=True)[0]
    q_hist = np.histogram(q.to_numpy(), bins, density=True)[0]

    # Avoid division by zero
    p_hist[p_hist == 0] = np.finfo(float).eps
    q_hist[q_hist == 0] = np.finfo(float).eps

    return entropy(p_hist, q_hist, base=2)


def create_internal_granger_matrix(ts: np.ndarray | pd.DataFrame, maxlag=3) -> np.ndarray | pd.DataFrame:
    from statsmodels.tsa.stattools import grangercausalitytests
    """
    Create a Granger causality matrix for all feature pairs within a single time series dataset.
    Each cell (i, j) in the matrix represents the Granger causality test result for feature i causing feature j.
    Reference: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
    """
    if isinstance(ts, pd.DataFrame):
        columns = ts.columns
        ts = ts.to_numpy()
    else:
        columns = None

    num_features = ts.shape[1]
    causality_matrix = np.zeros((num_features, num_features), dtype=np.float32)

    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                combined_data = np.column_stack((ts[:, i], ts[:, j]))
                result = grangercausalitytests(combined_data, maxlag=maxlag, verbose=False)
                # We are interested in any causality, so we take the minimum p-value over all lags up to maxlag
                # Options of 'ssr_chi2test' and 'params_ftest' are available. I'm not sure which is better.
                p_values = [result[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag + 1)]
                causality_matrix[i, j] = np.min(p_values)  # Choose the minimum p-value
    if columns is not None:
        return pd.DataFrame(causality_matrix, columns=columns, index=columns)
    else:
        return causality_matrix


def compare_time_by_granger_causality(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    # you may go crazy here (do whatever)
    matrix1 = create_internal_granger_matrix(df1)
    matrix2 = create_internal_granger_matrix(df2)
    # I chose KLD for fun. There's likely a better summary metric.
    return kld(matrix1, matrix2)


if __name__ == "__main__":
    # dataset1 and dataset2 are numpy arrays with shape (n_samples, n_features)
    columns = ["A", "B", "C", "D", "E", "F"]
    df1 = pd.DataFrame(np.random.rand(50000, 6), columns=columns)
    df2 = pd.DataFrame(np.random.rand(50000, 6), columns=columns)
    # create_internal_granger_matrix(df1)
    compare_time_by_granger_causality(df1, df2)
    pass
