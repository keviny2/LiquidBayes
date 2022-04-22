import os
import pyro
import pyro.distributions as pydist
import numpyro
import numpyro.distributions as numdist
import numpy as np
import jax.numpy as jnp
import torch

# from distributions import UniformDiscrete


# model definitions

def simple_model(data, cn_profiles, num_clones):
    """
    :param data: (n,) numpy array
    :param cn_profiles: (n, num_clones) numpy array
    :param num_clones: integer
    """
    omega = numpyro.sample('omega', numdist.Gamma(0.5, 0.001))
    tau = numpyro.sample('tau', numdist.Gamma(1, 1))

    probs = np.ones(num_clones) / np.sum(np.ones(num_clones))
    rho = numpyro.sample('rho', numdist.Dirichlet(probs))

    mu = omega * np.sum(cn_profiles * rho, axis=1)

    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=100, loc=mu, scale=tau), obs=data)

def one_additional_clone(data, cn_profiles, num_clones):
    omega = pyro.sample('omega', pydist.Gamma(0.5, 0.001))
    tau = pyro.sample('tau', pydist.Gamma(1, 1))
  
    probs = torch.ones(num_clones + 1) / (num_clones + 1)  # add additional clone cluster
    print(probs)
    rho = pyro.sample('rho', pydist.Dirichlet(probs))

    with pyro.plate('new clone', size=len(data)):
        new_clone_cn_profile = pyro.sample('new cn profile',
                                           UniformDiscrete(torch.FloatTensor([1,2,3,4,5,6,7,8,9]), torch.ones(9)/9))

    mu = omega * torch.sum(torch.cat((cn_profiles, new_clone_cn_profile[:, None]), dim=1) * rho,
                           dim=1)

    with pyro.plate('data', size=len(data)):
        pyro.sample('obs', pydist.StudentT(df=10, loc=mu, scale=tau), obs=data)
