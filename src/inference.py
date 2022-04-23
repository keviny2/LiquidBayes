import os
import pyro
import numpyro
from jax import random
import torch

from src.models import simple_model, one_additional_clone


def run_inference(model_type, data, cn_profiles, num_samples, num_warmup, iteration, target_accept_prob=0.95):
    
    if model_type == 'simple-model':
        model = simple_model
        return run_mcmc(model, model_type, data, cn_profiles, numpyro.infer.MCMC, numpyro.infer.NUTS, num_samples, num_warmup, iteration, target_accept_prob)
    if model_type == 'one-additional-clone':
        model = one_additional_clone
        return run_mcmc(model, model_type, data, cn_profiles, pyro.infer.mcmc.api.MCMC, pyro.infer.mcmc.NUTS, num_samples, num_warmup, iteration, target_accept_prob)
    if model_type == 'svi':
        model = svi

def run_mcmc(model, model_type, data, cn_profiles, inf_method, method, num_samples, num_warmup, iteration, target_accept_prob):
    if model_type == 'simple-model':
        sampler_obj = inf_method(method(model, target_accept_prob=target_accept_prob),
                                 num_warmup=num_warmup,
                                 num_samples=num_samples,
                                 progress_bar=False)
        sampler_obj.run(random.PRNGKey(iteration), data, cn_profiles, cn_profiles.shape[1])
    if model_type == 'one-additional-clone':
        data = torch.from_numpy(data)
        cn_profiles = torch.from_numpy(cn_profiles)
        pyro.set_rng_seed(int(iteration))
        sampler_obj = inf_method(method(model, target_accept_prob=target_accept_prob),
                                 warmup_steps=num_warmup,
                                 num_samples=num_samples,
                                 disable_progbar=True)
        sampler_obj.run(data, cn_profiles, cn_profiles.shape[1])
    
    return sampler_obj

