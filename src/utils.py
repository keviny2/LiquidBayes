import numpy as np
import pandas as pd
import string
from collections import defaultdict
from sklearn.mixture import GaussianMixture


def load_data(data_path, cn_profiles_path):
    data = np.genfromtxt(data_path, delimiter='\t')[:, -1]
    cn_profiles = np.genfromtxt(cn_profiles_path, delimiter='\t')[:, 3:]  # do not need genomic locations from bed file
    return data, cn_profiles

def save_results(path, sampler_obj, num_subclones):
    samples = pd.DataFrame.from_dict(sampler_obj.get_samples())  # get samples from inference
    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    rhos = pd.DataFrame(samples['rho'].to_list(), columns=clones, dtype=float) 
    samples.join(rhos).drop('rho', axis=1).describe().loc[['mean']].to_csv(path)  # write mean of each sample site to csv file

def remove_outliers(cn_config, data, cn_profiles, indices, vals):
    """
    given a CN configuration and dataset, remove values at indices which correspond to outliers
    Arguments:
        cn_config: tuple
        data: ndarray
        cn_profiles: ndarray
    Returns:
        dataset with outliers removed
    """
        
    # fit gmm
    X = np.array(vals[cn_config]).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2).fit(X)  # fit gmm
    labels = gmm.predict(X)  # get assignments for each observation
    
    if gmm.covariances_[0].squeeze() < gmm.covariances_[1].squeeze():
        labels = np.invert(labels.astype(bool))
    
    # remove outliers from data.obs
    outlier_idxs = np.array(indices[cn_config]).reshape(-1,1)[labels == 0]
    
    # set the outliers as nans and remove afterwards
    data[outlier_idxs] = np.nan
    cn_profiles[outlier_idxs] = np.nan
    
def preprocess_data(data, cn_profiles):

    # create dictionaries storing info for each CN configuration
    indices = defaultdict(list)
    vals = defaultdict(list)
    for n in range(len(data)):
        indices[tuple(cn_profiles[n])].append(n)  # append the index to data.obs
        vals[tuple(cn_profiles[n])].append(data[n])
        
    # remove outliers from each CN configuration
    for cn_config in list(vals.keys()):
        if len(vals[cn_config]) < 50:
            continue
        remove_outliers(cn_config, data, cn_profiles, indices, vals)

    nan_idxs = ~np.isnan(data)
    data = data[nan_idxs]
    cn_profiles = cn_profiles[nan_idxs, :]

    return data, cn_profiles
