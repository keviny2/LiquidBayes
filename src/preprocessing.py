import os
import subprocess
import numpy as np
import pandas as pd
import pysam
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import pyranges as pr

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
# only display errors, suppress warning messages
from rpy2.rinterface_lib.callbacks import logger
import logging
logger.setLevel(logging.ERROR)

from src.utils import get_random_string 



def get_reads(bam_file_path, chrs, bin_size, qual):
    if not os.path.exists(bam_file_path + '.bai'):
        print('Indexing {}'.format(bam_file_path))
        pysam.index(bam_file_path)

    readcount_path = f'temp/readcounts{get_random_string()}.wig'
    os.makedirs('temp', exist_ok=True)
    command = f"readCounter {bam_file_path} -c {chrs} -w {bin_size} -q {qual} > {readcount_path}"
    subprocess.run(command, shell=True, check=True)
    return readcount_path

def correct_reads(readcount_path, gc_path, map_path):
    hmmcopy = importr('HMMcopy')
    data = hmmcopy.wigsToRangedData(readcount_path, 
                                    gc_path,
                                    map_path)
    data = hmmcopy.correctReadcount(data)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        corrected_readcounts = robjects.conversion.rpy2py(data)[['chr', 'start', 'end', 'copy']]
    return corrected_readcounts

def intersect(corrected_readcounts, cn_profiles_path):
    cn_profiles = pd.read_csv(cn_profiles_path, sep='\t', header=None)

    # format column names for PyRanges
    corrected_readcounts.rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}, inplace=True)
    cn_profiles.columns = [str(num) for num in range(cn_profiles.shape[1])]
    cn_profiles.rename(columns={'0': 'Chromosome', '1': 'Start', '2': 'End'}, inplace=True)

    # intersect both ways
    corrected_readcounts_gr, cn_profiles_gr = pr.PyRanges(corrected_readcounts).sort(), pr.PyRanges(cn_profiles).sort()
    gr1 = corrected_readcounts_gr.intersect(cn_profiles_gr)
    gr2 = cn_profiles_gr.intersect(corrected_readcounts_gr)

    corrected_readcounts_intersected = pd.concat([gr1.df.astype({'Chromosome': int}), gr2.df.astype({'Chromosome': int})], axis=1).dropna()
    return corrected_readcounts_intersected[['copy']].to_numpy().squeeze(), corrected_readcounts_intersected.iloc[:, -3:].to_numpy().squeeze()

def preprocess_bam_file(bam_file_path, cn_profiles_path, chrs, bin_size, qual, gc, mapp):
    print('Processing .bam file')
    print('Getting readcounts')
    readcount_path = get_reads(bam_file_path, chrs, bin_size, qual)

    print('Correcting readcounts')
    corrected_readcounts = correct_reads(readcount_path, gc, mapp)

    print('Intersecting readcounts with CN profiles')
    print(cn_profiles_path)
    data, cn_profiles = intersect(corrected_readcounts, cn_profiles_path)

    # remove unnecessary file
    os.remove(readcount_path)

    return data, cn_profiles
    
def preprocess_cn_configs(data, cn_profiles):
    print('Preprocessing within copy number configurations')

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
