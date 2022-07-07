import os
import numpy as np
import numpyro
from jax import random

from src.models import simple


def run_inference(model,
                  data,
                  cn_profiles,
                  num_samples,
                  num_warmup,
                  iteration,
                  progress_bar,
                  target_accept_prob=0.95):

    if model in ['simple']:
        sampler_obj = numpyro.infer.MCMC(numpyro.infer.NUTS(eval(model.replace('-', '_')), target_accept_prob=target_accept_prob),  # convert '-' to '_' to match function name
                                         num_warmup=num_warmup,
                                         num_samples=num_samples,
                                         progress_bar=progress_bar)
        sampler_obj.run(random.PRNGKey(iteration), data, cn_profiles, cn_profiles.shape[1])

    return sampler_obj
