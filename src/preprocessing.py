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

from src.utils import get_random_string, _print



def get_reads(bam_file_path, chrs, bin_size, qual, verbose, temp_dir):
    """
    Run readCounter from hmmcopy_utils to get binned read counts
    Arguments:
        bam_file_path: a string
        chrs: a string
        bin_size: a string
        qual: a string
        verbose: a boolean
        temp_dir: a string
    Returns:
        Path to file containing binned read counts
    """
    if not os.path.exists(bam_file_path + '.bai'):
        _print('Indexing {}'.format(bam_file_path), verbose)
        pysam.index(bam_file_path)

    readcount_path = os.path.join(temp_dir, f'readcounts{get_random_string()}.wig')
    os.makedirs(temp_dir, exist_ok=True)
    command = f"readCounter {bam_file_path} -c {chrs} -w {bin_size} -q {qual} > {readcount_path}"
    subprocess.run(command, shell=True, check=True)
    return readcount_path

def correct_reads(readcount_path, gc_path, map_path):
    """
    Use HMMcopy package to perform gc and mappability correction
    Arguments:
        readcount_path: a string
        gc_path: a string
        map_path: a string
    Returns:
        Pandas dataframe with the corrected read counts
    """
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
    return corrected_readcounts_intersected[['copy']].to_numpy().squeeze(), corrected_readcounts_intersected.iloc[:, 4:].to_numpy().squeeze()

def preprocess_bam_file(bam_file_path, cn_profiles_path, chrs, bin_size, qual, gc, mapp, verbose, temp_dir):
    _print('Processing .bam file', verbose)
    _print('Getting readcounts', verbose)
    readcount_path = get_reads(bam_file_path, chrs, bin_size, qual, verbose, temp_dir)

    _print('Correcting readcounts', verbose)
    corrected_readcounts = correct_reads(readcount_path, gc, mapp)

    _print('Intersecting readcounts with CN profiles', verbose)
    data, cn_profiles = intersect(corrected_readcounts, cn_profiles_path)

    # remove unnecessary file
    os.remove(readcount_path)

    return data, cn_profiles
    
def remove_outliers(data, cn_profiles, verbose):
    """
    Remove outliers based on CN configuration - ex. (2,2,2), (2,3,2)
    Arguments:
        data: ndarray
        cn_profiles: ndarray
        verbose: bool
    Returns:
        Two ndarrays corresponding to original arguments data, cn_profiles with outliers filtered out
    """

    _print('Identifying and removing outliers based on copy number configurations', verbose)

    def remove_outliers_in_cn_config(cn_config, data, cn_profiles, indices, vals):
        """
        For a specific CN configuration, remove outliers by fitting a two component GMM
        Arguments:
            cn_config: tuple
            data: ndarray
            cn_profiles: ndarray
        """
            
        # fit gmm
        X = np.array(vals[cn_config]).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2).fit(X)  # fit gmm
        labels = gmm.predict(X)  # get assignments for each observation
        
        if gmm.covariances_[0].squeeze() < gmm.covariances_[1].squeeze():
            labels = np.invert(labels.astype(bool))
        
        # remove outliers from data.obs
        outlier_idxs = np.array(indices[cn_config]).reshape(-1,1)[labels == 0]
        
        # set the outliers to nan
        data[outlier_idxs] = np.nan
        cn_profiles[outlier_idxs] = np.nan

    # create dictionaries storing info for each CN configuration
    indices = defaultdict(list)
    vals = defaultdict(list)
    for n in range(len(data)):
        indices[tuple(cn_profiles[n, 3:])].append(n)  # don't need first 3 columns containing genomic position information
        vals[tuple(cn_profiles[n, 3:])].append(data[n])
        
    # remove outliers from each CN configuration
    for cn_config in list(vals.keys()):
        if len(vals[cn_config]) < 50:
            continue
        remove_outliers_in_cn_config(cn_config, data, cn_profiles, indices, vals)

    nan_idxs = ~np.isnan(data)
    data = data[nan_idxs]
    cn_profiles = cn_profiles[nan_idxs, :]

    return data, cn_profiles
