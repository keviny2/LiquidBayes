import os
import sys
import numpy as np
import pandas as pd
import string
import random


def save_results(path, sampler_obj, num_subclones, verbose):
    dct = sampler_obj.get_samples()
    clones = list(string.ascii_uppercase)[:2] + ['normal']
    rhos = pd.DataFrame(list(dct['rho']),columns=clones, dtype = float)
    dct.pop('rho')
    samples = pd.DataFrame.from_dict(dct)  # get samples from inference

    res_dir = os.path.dirname(path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if os.path.exists(path):
        _print('Overwriting {}'.format(os.path.basename(path)), verbose)
        os.remove(path)

    _print('Saving results', verbose)
    samples.join(rhos).describe().loc[['mean']].to_csv(path, index=False)  # write mean of each sample site to csv file

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
