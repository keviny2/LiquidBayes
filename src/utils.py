import numpy as np
import pandas as pd
import string
import random
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector


def load_data(data_path, cn_profiles_path):
    data = np.genfromtxt(data_path, delimiter='\t')[:, -1]
    cn_profiles = np.genfromtxt(cn_profiles_path, delimiter='\t')[:, 3:]  # do not need genomic locations
    return data, cn_profiles

def save_results(path, sampler_obj, num_subclones):
    samples = pd.DataFrame.from_dict(sampler_obj.get_samples())  # get samples from inference
    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    rhos = pd.DataFrame(samples['rho'].to_list(), columns=clones, dtype=float)
    samples.join(rhos).drop('rho', axis=1).describe().loc[['mean']].to_csv(path, index=False)  # write mean of each sample site to csv file

def get_random_string(length=10):
    """
    generate a random string; used to define unique file paths if running LiquidBayes in parallel
    """

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def import_R_pkgs(packnames):
    utils = rpackages.importr('utils')
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
