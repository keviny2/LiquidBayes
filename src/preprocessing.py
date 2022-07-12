import os
import subprocess
import pandas as pd
import pysam
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from rpy2 import robjects
import pybedtools as pbt

from src.utils import import_R_pkgs, get_random_string


def get_reads(bam_file_path, chrs, qual):
    if not os.path.exists(bam_file_path + '.bai'):
        print('Indexing {}'.format(bam_file_path))
        pysam.index(bam_file_path)

    # TODO: might want to modify so that output_file_path isn't hard coded
    readcount_path = 'extdata/temp/readcounts.wig' + get_random_string()
    subprocess.run(['../hmmcopy_utils/bin/readCounter',
                    '-c', chrs,
                    '-w', bin_size,
                    '-q', qual,
                    '>', readcount_path],
                    shell=True,
                    check=True)
    return readcount_path

def correct_reads(readcount_path):
    import_R_pkgs(('HMMcopy', 'dplyr'))
    data_full_path = 'extdata/temp/data_full.bed' + get_random_string()
    # TODO: figure out a way to have gc and map wig files not be hard coded?
    robjects.r('''
            copy <- wigsToRangedData(readcount_path, extdata/ref/b37.gc.wig, extdata/ref/b37.map.wig)
            normal_copy <- correctReadcount(copy)
            normal_copy <- normal_copy %>% dplyr::select(chr, start, end, copy)
            normal_copy$chr <- normal_copy$chr %>% as.numeric() %>% sort()
            write.table(normal_copy, {}, sep='\t', col.names = FALSE, row.names = FALSE)
            '''.format(data_full_path))
    return data_full_path

def intersect(data_full_path, cn_profiles_path):
    # define temporary files
    temp_data = 'extdata/temp/temp_data' + get_random_string() + '.bed'
    temp_cn = 'extdata/temp/temp_cn' + get_random_string() + '.bed'

    pbt.BedTool(data_full_path).intersect(cn_profiles_path).saveas(temp_data)
    pbt.BedTool(cn_profiles_path).intersect(data_full_path).saveas(temp_cn)

    bed = pd.read_csv(temp_data, sep='\t', header=None)
    cn_bed = pd.read_csv(temp_cn, sep='\t', header=None)

    df = pd.concat([bed, cn_bed], axis=1)
    df = df.dropna()

    # remove unnecessary files
    os.remove(temp_data)
    os.remove(temp_cn)

    return df.iloc[:, :4], df.iloc[:, 4:]

def preprocess_bam_file(bam_file_path, cn_profiles_path, chrs, bin_size, qual):
    print('Getting readcounts')
    readcount_path = get_reads(bam_file_path, chrs, qual)

    print('Correcting readcounts')
    data_full_path = correct_reads(readcount_path)

    print('Intersecting readcounts with CN profiles')
    data, cn_profiles = intersect(data_full_path, cn_profiles_path)

    # remove unnecessary files
    os.remove(readcount_path)
    os.remove(data_full_path)

    return data, cn_profiles
    
def preprocess_from_cn_configs(data, cn_profiles):

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
