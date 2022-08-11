import os
import numpyro
import numpyro.distributions as numdist
import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey
import traceback
import pymc as pm
import aesara.tensor as at

def cn(data, cn_profiles, num_clones):
    """
    :param data: (n,) numpy array
    :param cn_profiles: (n, num_clones) numpy array
    :param num_clones: integer
    """
    rho = numpyro.sample('rho', numdist.Dirichlet(jnp.ones(num_clones)))
    mu = jnp.log(jnp.sum(cn_profiles*rho, axis=1)) - jnp.log(jnp.mean(jnp.sum(cn_profiles*rho, axis=1)))
    tau = numpyro.sample('tau', numdist.InverseGamma(3, 1))
    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=4, loc=mu, scale=tau), obs=data)

def one_more_clone(data, cn_profiles, num_clones, prob):
    """
    Infer tf for one additional subclone in rho with data (n,) and cn_profiles (n, num_clones)
    :param data: (n,) numpy array
    :param cn_profiles: (n, num_clones) numpy array
    :param num_clones: integer
    :returns: pymc model
    """
    one_addition_model = pm.Model()

    with one_addition_model:
        rho = pm.Dirichlet('rho', at.ones(cn_profiles.shape[1] + 1))
        new_cn_col = pm.DiscreteUniform('new_clone_cn', lower = 1, upper = 8, shape = len(data))

        new_cn_profiles = at.concatenate((cn_profiles, new_cn_col.reshape([len(data), 1])), axis = 1)
        mu = at.log(at.sum(new_cn_profiles*rho, axis=1)) - at.log(at.mean(at.sum(new_cn_profiles*rho, axis=1)))
        tau = pm.InverseGamma('tau', alpha=3, beta=1)
        obs = pm.StudentT('obs', nu=4, mu=mu, sigma=tau, observed=data)
        step=[pm.NUTS([rho, tau], target_accept=prob), pm.Metropolis([new_cn_col])]
    return one_addition_model, step
                                
