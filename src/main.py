import numpy as np
import pandas as pd
import numpyro

from inference import run_inference


def run(model, num_samples, num_warmup, seed, data, cn_profiles, output):
    # load data and cn_profiles
    data_np = np.genfromtxt(data, delimiter='\t')[:, -1]
    cn_profiles_np = np.genfromtxt(cn_profiles, delimiter=',')
    sampler_obj = run_inference(model,
                                data_np.squeeze(),
                                cn_profiles_np.squeeze(),
                                num_samples,
                                num_warmup,
                                int(seed))

    summary_dict = numpyro.diagnostics.summary(sampler_obj.get_samples(group_by_chain=True))
    summary = pd.DataFrame.from_dict(summary_dict)
    summary = summary.loc[['mean']]  # only keep mean

    # drop unwanted columns and rows and rename
    summary.to_csv(output, index=False)
