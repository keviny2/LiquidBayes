import os
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceMeanField_ELBO
import pyro.optim as optim
import numpyro
from jax import random
from tqdm import trange

from src.models import simple_model, one_additional_clone, version_2, version_2_guide, version_2_mcmc, version_2_pymc


def run_inference(model,
                  data,
                  cn_profiles,
                  num_samples,
                  num_warmup,
                  iteration,
                  progress_bar,
                  target_accept_prob=0.95):

    if model in ['simple-model']:
        sampler_obj = numpyro.infer.MCMC(numpyro.infer.NUTS(eval(model.replace('-', '_')), target_accept_prob=target_accept_prob),  # convert '-' to '_' to match function name
                                         num_warmup=num_warmup,
                                         num_samples=num_samples,
                                         progress_bar=progress_bar)
        sampler_obj.run(random.PRNGKey(iteration), data, cn_profiles, cn_profiles.shape[1])

    if model in ['version-2-pymc']:
        sampler_obj = version_2_pymc(data, cn_profiles, num_clones, num_samples)

    if model in ['version-2-mcmc']:
        sampler_obj = pyro.infer.MCMC(pyro.infer.NUTS(eval(model.replace('-', '_')), target_accept_prob=target_accept_prob),  # convert '-' to '_' to match function name
                                      warmup_steps=num_warmup,
                                      num_samples=num_samples,
                                      disable_progbar=not progress_bar)
        sampler_obj.run(torch.from_numpy(data), torch.from_numpy(cn_profiles), cn_profiles.shape[1])

    # BUG: version-2-vi still not working
    if model in ['version-2-vi']:
        sampler_obj = SVI(version_2,
                          version_2_guide,
                          optim.ClippedAdam({"lr": .0001,
                                             "betas": (0.95, 0.999),
                                             "lrd": 0.1**(1/num_samples)}),
                          loss=TraceMeanField_ELBO(num_particles=5))

        with trange(num_samples) as t:
            for i in t:
                elbo = sampler_obj.step(torch.from_numpy(data), torch.from_numpy(cn_profiles), cn_profiles.shape[1])
                t.set_postfix(loss=elbo)

        predictive = Predictive(version_2, guide=version_2_guide, num_samples=1000)
        svi_samples = {k: v.reshape(1000).detach().cpu().numpy()
                       for k, v in predictive(torch.from_numpy(data), torch.from_numpy(cn_profiles), cn_profiles.shape[1]).items()
                       if k != "obs"}

        for site, values in summary(svi_samples).items():
            print("Site: {}".format(site))
            print(values, "\n")
        
    return sampler_obj
