import os
import sys
import numpy as np
import pandas as pd
import string
import random
import arviz as az
from scipy import stats


def save_results(path, sampler_obj, num_subclones, verbose):
    res_dir = os.path.dirname(path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if os.path.exists(path):
        _print('Overwriting {}'.format(os.path.basename(path)), verbose)
        os.remove(path)

    _print('Saving results', verbose)
    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    samples = sampler_obj.get_samples()
    rhos = pd.DataFrame(list(samples['rho']),columns=clones, dtype = float)
    samples.pop('rho')
    samples = pd.DataFrame.from_dict(samples)  
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
