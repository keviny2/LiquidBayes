import os
import pyro
import pyro.distributions as pyrodist
import pyro.poutine as poutine
import numpyro
import numpyro.distributions as numdist
import numpy as np
import jax.numpy as jnp
import torch
from torch.distributions import constraints


# model definitions

def simple_model(data, cn_profiles, num_clones):
    """
    :param data: (n,) numpy array
    :param cn_profiles: (n, num_clones) numpy array
    :param num_clones: integer
    """
    omega = numpyro.sample('omega', numdist.Gamma(1, 1))
    tau = numpyro.sample('tau', numdist.InverseGamma(3, 1))
    delta = numpyro.sample('delta', numdist.Normal(0, 1))

    probs = np.ones(num_clones) / jnp.sum(np.ones(num_clones))
    rho = numpyro.sample('rho', numdist.Dirichlet(probs, validate_args=False))

    mu = jnp.log2(omega * jnp.sum(cn_profiles * rho, axis=1) + delta)

    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=5000, loc=mu, scale=tau), obs=data)

def ichor(data, cn_profiles, num_clones):
    """
    :param data: (n,) numpy array
    :param cn_profiles: (n, num_clones) numpy array
    :param num_clones: integer
    """
    rho = numpyro.sample('rho', numdist.Dirichlet(jnp.ones(num_clones)))
    mu = jnp.log(jnp.sum(cn_profiles*rho, axis=1)) - jnp.log(jnp.mean(jnp.sum(cn_profiles*rho, axis=1)))
    tau = numpyro.sample('tau', numdist.InverseGamma(1, 1))
    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=4, loc=mu, scale=tau), obs=data)

def version_2_mcmc(data, cn_profiles, num_clones):
    """
    :param data: (n,) torch tensor
    :param cn_profiles: (n, num_clones) torch tensor
    :param num_clones: integer
    """

    omega = numpyro.sample('omega', numdist.Gamma(1, 1))
    delta = numpyro.sample('delta', numdist.Normal(0, 1))
    rho = numpyro.sample('rho', numdist.Dirichlet(jnp.ones(num_clones)))
    
    mu_tilde = {}
    max_cn = int(np.amax(cn_profiles.flatten()))
    for c in range(max_cn+1):
        mu_tilde[c] = numpyro.sample('mu_tilde_{}'.format(c), numdist.Normal(c, .001)) 

    mu = omega * jnp.array([jnp.sum(jnp.array([mu_tilde[int(c)] for c in cn_profile])) for cn_profile in cn_profiles]) + delta
    tau = numpyro.sample('tau', numdist.InverseGamma(3, 1))
    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=4, loc=mu, scale=tau), obs=data)

@poutine.scale(scale=1.0/5000)
def version_2(data, cn_profiles, num_clones):
    """
    :param data: (n,) torch tensor
    :param cn_profiles: (n, num_clones) torch tensor
    :param num_clones: integer
    """

    probs = torch.ones(num_clones) / torch.sum(torch.ones(num_clones))
    rho = pyro.sample('rho', pyrodist.Dirichlet(probs))
    
    # create random variables for each copy-number state
    mu_tilde = {}
    max_cn = int(torch.amax(cn_profiles.flatten()))
    for c in range(max_cn+1):
        if c <= 8:
            mu_tilde[c] = pyro.sample('mu_tilde_{}'.format(c), pyrodist.Normal(c*.2-.7, .035)) 
        else:
            mu_tilde[c] = 1

    mu = torch.tensor([torch.sum(torch.tensor([mu_tilde[int(c)] for c in cn_profile])) for cn_profile in cn_profiles])
    tau = torch.tensor([0.1*mu_i + 0.2 if mu_i < 2 else 0.5 for mu_i in mu]) # NOTE: temporary values for now

    with pyro.plate('data', size=len(data), subsample_size=5) as ind:
        pyro.sample('obs', pyrodist.StudentT(df=5000, loc=mu[ind], scale=tau[ind]), obs=data[ind])

@poutine.scale(scale=1.0/5000)
def version_2_guide(data, cn_profiles, num_clones):
    probs = pyro.param('probs',
                       torch.ones(num_clones) / torch.sum(torch.ones(num_clones)),
                       constraint=constraints.positive)

    rho = pyro.sample('rho', pyrodist.Dirichlet(probs)) 

    # create random variables for each copy-number state
    mu_tilde = {}
    phi_c = {}
    max_cn = int(torch.amax(cn_profiles.flatten()))
    for c in range(max_cn+1):
        if c <= 8:
            phi_c[c] = pyro.param('phi_{}'.format(c), pyrodist.Normal(c*.2-.7, .01))
            mu_tilde[c] = pyro.sample('mu_tilde_{}'.format(c), pyrodist.Normal(phi_c[c], .035)) 
        else:
            mu_tilde[c] = 1

    mu = torch.tensor([torch.sum(torch.tensor([mu_tilde[int(c)] for c in cn_profile])) for cn_profile in cn_profiles])
    tau = torch.tensor([0.1*mu_i + 0.2 if mu_i < 2 else 0.5 for mu_i in mu]) # NOTE: temporary values for now
