import os
import aesara.tensor as at
import numpy as np
import numpyro
import pymc as pm
from numpyro.infer import MCMC, NUTS
from jax import random

from src.utils import _print
from src.models import cn, cn_snv, one_more_clone


def run_inference(model,
                  data,
                  cn_profiles,
                  counts,
                  num_samples,
                  num_warmup,
                  iteration,
                  progress_bar,
                  verbose,
                  target_accept_prob=0.95):

    _print('Performing inference using {} model'.format(model), verbose)
    if model == 'cn':
        sampler_obj = numpyro.infer.MCMC(numpyro.infer.NUTS(eval(model.replace('-', '_')), target_accept_prob=target_accept_prob),  # convert '-' to '_' to match function name
                                         num_warmup=num_warmup,
                                         num_samples=num_samples,
                                         progress_bar=progress_bar)
        sampler_obj.run(random.PRNGKey(iteration), data, cn_profiles, cn_profiles.shape[1])
        return sampler_obj

    if model == 'cn_snv':
        np.savetxt('data.tsv', data, delimiter='\t')
        np.savetxt('cn_profiles.tsv', cn_profiles, delimiter='\t')
        np.savetxt('counts.tsv', counts, delimiter='\t')
        sampler_obj = numpyro.infer.MCMC(numpyro.infer.NUTS(eval(model.replace('-', '_')), target_accept_prob=target_accept_prob),  # convert '-' to '_' to match function name
                                         num_warmup=num_warmup,
                                         num_samples=num_samples,
                                         progress_bar=progress_bar)
        sampler_obj.run(random.PRNGKey(iteration), data, cn_profiles, counts, cn_profiles.shape[1])
        return sampler_obj
