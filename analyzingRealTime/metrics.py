import numpy as np
import pandas as pd

#import sklearn.metrics as skm

#Use these three
#kld (Kullback-Leibler Divergence). Created in this file.
#skm.mutual_info_score(data1, data2) #Entropy-based. Using instead of making a function of autoencoder size to score.
#scipy.stats.wasserstein_distance # Earth Mover's Distance, used in WGANs


#And select one from here
#skm.pairwise.cosine_similarity
#L2 norm: np.linalg.norm(x,ord=2) # Euclidean
#L1 norm: np.linalg.norm(x,ord=1) # Manhattan


def kld(p:pd.DataFrame, q:pd.DataFrame, bins=35)->float:
    from scipy.stats import entropy
    """
    Compute the Kullback-Leibler Divergence using scipy's entropy function. KLD(p||q).
    This is an assymetric measure. KLD(p||q) can be understood as the amount of information lost when q is used to approximate p.
    So, a lower result implies q is a better approximation of p.
    If there's time, I'll partition PDFs by sine/cosine pairs and each categorical feature. 
    There are pros and cons to partitioning and this holistic approach.
    Don't worry about that though, just plop in dataframes and expect a float to return.
    """
    p_hist = np.histogram(p.to_numpy(), bins, density=True)[0]
    q_hist = np.histogram(q.to_numpy(), bins, density=True)[0]

    # Avoid division by zero
    p_hist[p_hist == 0] = np.finfo(float).eps
    q_hist[q_hist == 0] = np.finfo(float).eps

    return entropy(p_hist, q_hist, base=2)

