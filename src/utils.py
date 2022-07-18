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
    #dct = sampler_obj.get_samples()
    #print(len(dct))
    #dct = {k:np.asarray(v) for k,v in dct.items()}
    #data = np.array(list(dct.items()), dtype=object)
    #np.save('./result.npy', data)
    dct = sampler_obj.get_samples()
    clones = list(string.ascii_uppercase)[:2] + ['normal']
    rhos = pd.DataFrame(list(dct['rho']),columns=clones, dtype = float)
    dct.pop('rho')
    samples = pd.DataFrame.from_dict(dct)  # get samples from inference

    res_dir = os.path.dirname(path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if os.path.exists(path):
        print('Overwriting {}'.format(os.path.basename(path)))
        os.remove(path)

    print('Saving results')
    samples.join(rhos).describe().loc[['mean']].to_csv(path, index=False)  # write mean of each sample site to csv file

def get_random_string(length=10):
    """
    generate a random string - used to define unique file paths if running LiquidBayes in parallel
    """

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
