import os
import numpy as np
import pandas as pd
import pyro
import numpyro
import string

from src.inference import run_inference


def run(input_path,
        cn_profiles_path,
        output,
        model,
        num_samples,
        num_warmup,
        seed,
        progress_bar):

    # load data and cn_profiles in ndarray
    data = np.genfromtxt(input_path, delimiter='\t')[:, -1]
    cn_profiles = np.genfromtxt(cn_profiles_path, delimiter='\t')[:, 3:]  # do not need genomic locations from bed file

    # perform inference
    sampler_obj = run_inference(model,
                                data.squeeze(),
                                cn_profiles.squeeze(),
                                num_samples,
                                num_warmup,
                                int(seed),
                                progress_bar)

    # postprocessing 
    samples = pd.DataFrame.from_dict(sampler_obj.get_samples())  # get samples from inference
    num_subclones = cn_profiles.shape[1]-1  # cn_profiles includes normal CN profile, so minus 1
    clones = list(string.ascii_uppercase)[:num_subclones] + ['normal']
    rhos = pd.DataFrame(samples['rho'].to_list(), columns=clones, dtype=float) 
    samples.join(rhos).drop('rho', axis=1).describe().loc[['mean']].to_csv(output)  # write mean of each sample site to csv file
