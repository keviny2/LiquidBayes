import os
import aesara.tensor as at
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC
from jax import random
import pymc as pm
from src.models import simple, one_more_clone


def run_inference(model,
                  data,
                  cn_profiles,
                  num_samples,
                  num_warmup,
                  iteration,
                  progress_bar,
                  target_accept_prob=0.95):

    print('Performing inference using {} model'.format(model))
    if model == 'one-more-clone':
        r = np.random.RandomState(iteration)

        one_addition_model,step = one_more_clone(data, cn_profiles, cn_profiles.shape[1], target_accept_prob)
        samples = pm.sample(draws=num_samples, tune=num_warmup, step=step, random_seed=r,progressbar=progress_bar, model=one_addition_model)
        return samples
    elif model == 'simple':
        sampler_obj = MCMC(NUTS(eval(model.replace('-', '_')), target_accept_prob=target_accept_prob),
                                        num_warmup=num_warmup,
                                        num_samples=num_samples,
                                        progress_bar=progress_bar)
        sampler_obj.run(random.PRNGKey(iteration), data, cn_profiles, cn_profiles.shape[1])

        return sampler_obj
