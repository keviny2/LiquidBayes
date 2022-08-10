import os
import sys
import numpy as np
import pandas as pd
import string
import random
import arviz as az


def load_data(data_path, cn_profiles_path):
    print('Processing .bed file')
    data = np.genfromtxt(data_path, delimiter='\t')[:, -1]
    cn_profiles = np.genfromtxt(cn_profiles_path, delimiter='\t')[:, 3:]  # do not need genomic locations
    return data, cn_profiles

def save_results(model, path, sampler_obj, num_subclones):

    res_dir = os.path.dirname(path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if os.path.exists(path):
        print('Overwriting {}'.format(os.path.basename(path)))
        os.remove(path)

    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    print('Saving results')
    if model == 'one-more-clone':
        arr = sampler_obj.posterior.new_clone_cn.to_numpy()
        np.save('./inferred_new_cn_profile.npy', arr)
        df = az.summary(sampler_obj, kind="stats")[-5:].T.head(1)
        df.columns = clones + ['tau']
        df.to_csv(path, index=False)
    elif model == 'simple':
        dct = sampler_obj.get_samples()
        rhos = pd.DataFrame(list(dct['rho']),columns=clones, dtype = float)
        dct.pop('rho')
        samples = pd.DataFrame.from_dict(dct)  
        samples.join(rhos).describe().loc[['mean']].to_csv(path, index=False)  
        
def get_random_string(length=10):
    """
    generate a random string - used to define unique file paths if running LiquidBayes in parallel
    """

    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
