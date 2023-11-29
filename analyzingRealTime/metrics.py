import sys
import scipy
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from dtaidistance import dtw
from sklearn import metrics
from typing import Callable
import warnings
warnings.filterwarnings("ignore")

###################################################################################################################

# Load data
data = pd.read_csv("al/preprocessed_old/times_for_static_data.csv")

DEBUG = sys.gettrace() is not None
NSAMPLES_PER_HOME = 1000 if DEBUG else "all"
NHOMES = 3 #if DEBUG else "all"

if NHOMES != "all":
    unique_ids = data["Person"].unique()
    data = data[data["Person"].isin(unique_ids[:NHOMES])]
if NSAMPLES_PER_HOME != "all":
    data = data.groupby("Person").head(NSAMPLES_PER_HOME)

# Drop unneeded columns
data = data[["DateTime", "Person", "SecondsCos", "SecondsSin", "DoyCos", "DoySin", "Sedentary"]]
data = data.set_index("DateTime")
ts_list = {}

# Create TS for each person
for person in data.Person.unique():
    ts_list[person] = data[data["Person"] == person].drop(["Person"], axis=1)

# Collected Measures
n = len(ts_list)
num_features = len(ts_list[1].columns)
measures_list = {}
###################################################################################################################
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

def matrix_from_1d_measure(
        multivariate_time_series: np.ndarray | pd.DataFrame,
        measure_func:Callable[[np.ndarray,np.ndarray,...],float],
        skip_same=False,
        reshape = True,
        **measure_func_kwargs
) -> np.ndarray | pd.DataFrame:
    if isinstance(multivariate_time_series, pd.DataFrame):
        columns = multivariate_time_series.columns
        multivariate_time_series = multivariate_time_series.to_numpy()
    else:
        columns = None

    num_features = multivariate_time_series.shape[1]
    measure_matrix = np.zeros((num_features, num_features), dtype=np.float32)

    for i in range(num_features):
        for j in range(num_features):
            if skip_same and i==j:
                continue
            if reshape == True:
                measure_matrix[i, j] = measure_func(
                    multivariate_time_series[:, i].reshape(1,-1),
                    multivariate_time_series[:, j].reshape(1,-1),
                    **measure_func_kwargs
                )
            else:
                measure_matrix[i, j] = measure_func(
                    multivariate_time_series[:, i],
                    multivariate_time_series[:, j],
                    **measure_func_kwargs
                )
    if columns is not None:
        return pd.DataFrame(measure_matrix, columns=columns, index=columns)
    else:
        return measure_matrix

def granger_causality_result(feature1:np.ndarray, feature2:np.ndarray, maxlag=3) -> float:
    combined_data = np.column_stack((feature1, feature2))
    result = grangercausalitytests(combined_data, maxlag=maxlag, verbose=False)
    # We are interested in any causality, so we take the minimum p-value over all lags up to maxlag
    # Options of 'ssr_chi2test' and 'params_ftest' are available. I'm not sure which is better.
    p_values = [result[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag + 1)]
    return np.min(p_values)  # Choose the minimum p-value

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
#####################################################################################################################
#Dynamic Time Warping - Direct Comparison Across Homes

DTW_matrix = np.zeros((n, n), dtype=np.float32)
temp = np.zeros((num_features, num_features), dtype=np.float32)
for i, person1 in enumerate(ts_list.keys()):
    for j, person2 in enumerate(ts_list.keys()):
        if i != j:
            ts1 = ts_list[person1].to_numpy()
            ts2 = ts_list[person2].to_numpy()
            temp = dtw.distance_matrix_fast([ts1[:,i].reshape(1,-1) for i in range(num_features)] +
                                            [ts2[:,j].reshape(1,-1) for j in range(num_features)])
            temp = temp[num_features:,0:num_features]
            DTW_matrix[i, j] = np.linalg.norm(temp)  #Frobenius norm

measures_list["DTW - Between"] = DTW_matrix

# Dynamic Time Warping - Comparison Within Homes
DTW_internal_matrices = {}
DTW_matrix = np.zeros((n, n), dtype=np.float32)

for i, person in enumerate(ts_list.keys()):
    ts = ts_list[person].to_numpy()
    temp = dtw.distance_matrix_fast([ts[:, i].reshape(1, -1) for i in range(num_features)])
    DTW_internal_matrices[person] = pd.DataFrame(temp)

for i, person1 in enumerate(ts_list.keys()):
    for j, person2 in enumerate(ts_list.keys()):
        DTW_matrix[i, j] = kld(DTW_internal_matrices[person1], DTW_internal_matrices[person2])

measures_list["DTW - Within"] = DTW_matrix

#####################################################################################################################
# Kullback-Leibler Divergence

kld_matrix = np.zeros((n, n), dtype=np.float32)
for i, person1 in enumerate(ts_list.keys()):
    for j, person2 in enumerate(ts_list.keys()):
        kld_matrix[i, j] = kld(ts_list[person1],ts_list[person2])

measures_list["KLD"] = kld_matrix
#####################################################################################################################
# Mutual Information Score
MIS_internal_matrices = {}
MIS_matrix = np.zeros((n, n), dtype=np.float32)
for i, person in enumerate(ts_list.keys()):
    MIS_internal_matrices[person] = matrix_from_1d_measure(ts_list[person], metrics.mutual_info_score, reshape=False)

for i, person1 in enumerate(ts_list.keys()):
    for j, person2 in enumerate(ts_list.keys()):
        MIS_matrix[i, j] = kld(MIS_internal_matrices[person1],MIS_internal_matrices[person2])

measures_list["Mutual Information Score"] = MIS_matrix
#####################################################################################################################
# Earth Mover's Distance

EM_matrix = np.zeros((n, n), dtype=np.float32)
temp = np.zeros((num_features, num_features), dtype=np.float32)
for i, person1 in enumerate(ts_list.keys()):
    for j, person2 in enumerate(ts_list.keys()):
        df1 = ts_list[person1]
        df2 = ts_list[person2]
        for s, feature1 in enumerate(df1.columns):
            for t, feature2 in enumerate(df2.columns):
                temp[s,t] = scipy.stats.wasserstein_distance(df1[feature1],df2[feature2])
                EM_matrix[i,j] = np.linalg.norm(temp) #Frobenius norm

measures_list["Earth Mover's Distance"] = EM_matrix
###################################################################################################################
# Granger Causality

gc_internal_matrices = {}
for i, person in enumerate(ts_list.keys()):
    gc_internal_matrices[person] = create_internal_granger_matrix(ts_list[person])

# Find Granger Causality Matrix
GC_matrix = np.zeros((n, n), dtype=np.float32)
for i, person1 in enumerate(ts_list.keys()):
    for j, person2 in enumerate(ts_list.keys()):
        GC_matrix[i, j] = kld(gc_internal_matrices[person1],gc_internal_matrices[person2])

measures_list["Granger Causality"] = GC_matrix
##################################################################################################################
# Cosine Similarity

cos_internal_matrices = {}
for i, person in enumerate(ts_list.keys()):
    cos_internal_matrices[person] = matrix_from_1d_measure(ts_list[person],
                                                           metrics.pairwise.cosine_similarity, reshape=True)

cos_matrix = np.zeros((n, n), dtype=np.float32)
for i, person1 in enumerate(ts_list.keys()):
    for j, person2 in enumerate(ts_list.keys()):
        cos_matrix[i, j] = kld(cos_internal_matrices[person1],cos_internal_matrices[person2])

measures_list["Cosine Similarity"] = cos_matrix
##################################################################################################################
# L2 norm: np.linalg.norm(x,ord=2) # Euclidean
# L1 norm: np.linalg.norm(x,ord=1) # Manhattan

##################################################################################################################
# Summarize results for pairs of homes
first_index = np.tile(range(n), n)
second_index = np.repeat(range(n), n)

for i, measure in enumerate(measures_list):
    if i == 0:
        flatten_matrix = measures_list[measure].flatten("F")
        summary = pd.DataFrame(flatten_matrix, index=[first_index, second_index])
    else:
        summary[measure] = measures_list[measure].flatten("F")

summary.columns = measures_list.keys()
