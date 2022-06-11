import os
import pyro
import pyro.distributions as numdist
import numpyro
import numpyro.distributions as numdist
import numpy as np
import jax.numpy as jnp
import torch


# model definitions

def simple_model(data, cn_profiles, num_clones):
    """
    :param data: (n,) numpy array
    :param cn_profiles: (n, num_clones) numpy array
    :param num_clones: integer
    """
    omega = numpyro.sample('omega', numdist.Gamma(1, 1))
    tau = numpyro.sample('tau', numdist.Gamma(1, 1))
    delta = numpyro.sample('delta', numdist.Normal(0, 1))

    probs = np.ones(num_clones) / np.sum(np.ones(num_clones))
    rho = numpyro.sample('rho', numdist.Dirichlet(probs, validate_args=False))

    mu = omega * np.sum(cn_profiles * rho, axis=1) + delta

    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=100, loc=mu, scale=tau), obs=data)

def one_additional_clone(data, cn_profiles, num_clones):
    omega = numpyro.sample('omega', numdist.Gamma(0.5, 0.001))
    tau = numpyro.sample('tau', numdist.Gamma(1, 1))
  
    probs = np.ones(num_clones + 1) / (num_clones + 1)  # add additional clone cluster
    rho = numpyro.sample('rho', numdist.Dirichlet(probs))

    with numpyro.plate('new clone', size=len(data)):
        new_clone_cn_profile = numpyro.sample('new cn profile',
                                              numdist.DiscreteUniform(1,9),
                                              infer={"enumerate": "parallel"})

    print('cn_profiles.shape:\n', cn_profiles.shape)
    print('cn_profiles:\n', cn_profiles)
    print('new_clone_cn_profile.shape:\n', new_clone_cn_profile.shape)
    print('new_clone_cn_profile:\n', new_clone_cn_profile)
    mu = omega * np.sum(np.concatenate((cn_profiles, new_clone_cn_profile[:, None]), axis=1) * rho, axis=1)

    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=10, loc=mu, scale=tau), obs=data)
