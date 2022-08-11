import os
import numpy as np
import numpyro
from jax import random

from src.models import cn
from src.utils import _print


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
    if model in ['cn']:
        sampler_obj = numpyro.infer.MCMC(numpyro.infer.NUTS(eval(model.replace('-', '_')), target_accept_prob=target_accept_prob),  # convert '-' to '_' to match function name
                                         num_warmup=num_warmup,
                                         num_samples=num_samples,
                                         progress_bar=progress_bar)
        sampler_obj.run(random.PRNGKey(iteration), data, cn_profiles, cn_profiles.shape[1])

    return sampler_obj
