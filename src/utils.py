import os
import sys
import numpy as np
import pandas as pd
import string
import random


def load_data(data_path, cn_profiles_path):
    print('Processing .bed file')
    data = np.genfromtxt(data_path, delimiter='\t')[:, -1]
    cn_profiles = np.genfromtxt(cn_profiles_path, delimiter='\t')[:, 3:]  # do not need genomic locations
    return data, cn_profiles

def save_results(path, sampler_obj, num_subclones):
    samples = pd.DataFrame.from_dict(sampler_obj.get_samples())  # get samples from inference
    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    rhos = pd.DataFrame(samples['rho'].to_list(), columns=clones, dtype=float)

    res_dir = os.path.dirname(path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if os.path.exists(path):
        print('Overwriting {}'.format(os.path.basename(path)))
        os.remove(path)

    print('Saving results')
    samples.join(rhos).drop('rho', axis=1).describe().loc[['mean']].to_csv(path, index=False)  # write mean of each sample site to csv file

def get_random_string(length=10):
    """
    generate a random string - used to define unique file paths if running LiquidBayes in parallel
    """

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def get_path_to(relative_path, path_to_file=__file__):
    """
    this function covers the case when LiquidBayes is executed from another directory, but still need to get paths to directories within LiquidBayes
    """
    dir_path = os.path.dirname(os.path.realpath(path_to_file))
    return os.path.join(dir_path, relative_path)
    
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
