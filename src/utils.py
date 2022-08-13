import os
import sys
import numpy as np
import pandas as pd
import string
import random
import arviz as az
from scipy import stats


def save_results(model, path, sampler_obj, num_subclones, verbose):
    res_dir = os.path.dirname(path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if os.path.exists(path):
        _print('Overwriting {}'.format(os.path.basename(path)), verbose)
        os.remove(path)

    _print('Saving results', verbose)
    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    if model == 'one-more-clone':
        arr = sampler_obj.posterior.new_clone_cn.to_numpy()
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
        res = stats.mode(arr, keepdims=True)[0]   ### Getting the mode across chains and all samples
        print(res.shape)
        df_cn = pd.DataFrame(res, columns=[f"Inferred_cn_profile[{i+1}]" for i in range(res.shape[1])])
        df = az.summary(sampler_obj, kind="stats")[-5:].T.head(1)
        df.columns = clones + ['tau']
        result = pd.concate([df, df_cn], axis=1)
        result.to_csv(path, index=False)
    elif model == 'cn':
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

def _print(message, verbose):
    if verbose:
        print(message)
