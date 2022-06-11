import numpy as np
import pandas as pd
import numpyro

from src.inference import run_inference


def run(input_path, cn_profiles_path, output, model, num_samples, num_warmup, seed):
    # load data and cn_profiles
    data = np.genfromtxt(input_path, delimiter='\t')[:, -1]
    cn_profiles = np.genfromtxt(cn_profiles_path, delimiter='\t')[:, 3:]  # do not need genomic locations from bed file
    sampler_obj = run_inference(model,
                                data.squeeze(),
                                cn_profiles.squeeze(),
                                num_samples,
                                num_warmup,
                                int(seed))

    summary_dict = numpyro.diagnostics.summary(sampler_obj.get_samples(group_by_chain=True))
    summary = pd.DataFrame.from_dict(summary_dict)
    summary = summary.loc[['mean']]  # only keep mean

    # drop unwanted columns and rows and rename
    summary.to_csv(output, index=False)
