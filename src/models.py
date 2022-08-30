import os
import numpyro
import numpyro.distributions as numdist
import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey
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

def cn_snv(data, cn_profiles, counts, num_clones):
    """
    :param data: (n,) numpy array
    :param cn_profiles: (n, num_clones) numpy array
    :param counts: (L, 2+K) numpy array - L=length of intersection of SNV positions, 2+K=ref and alt counts for liquid biopsy and num clones
    :param num_clones: integer
    """
    rho = numpyro.sample('rho', numdist.Dirichlet(jnp.ones(num_clones)))
    mu = jnp.log(jnp.sum(cn_profiles*rho, axis=1)) - jnp.log(jnp.mean(jnp.sum(cn_profiles*rho, axis=1)))
    tau = numpyro.sample('tau', numdist.InverseGamma(3, 1))
    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=4, loc=mu, scale=tau), obs=data)

    # xi = jnp.sum(counts[:, 2:]*rho, axis=1)
    # d = jnp.sum(counts[:, :2], axis=1)
    # b = counts[:, 1]
    # with numpyro.plate('snv', size=len(xi)):
    #     numpyro.sample('snv_obs', numdist.Binomial(d, xi), obs=b)
    n = len(counts)
    counts[:, 2:] = (counts[:, 2:] * (n-1) + .5) / n  # transform clone level counts for BetaProportion distribution
    d = jnp.sum(counts[:, :2], axis=1)
    b = counts[:, 1]
    nu = counts[:, 2:]
    with numpyro.plate('snv', size=len(nu)):
        theta = jnp.ones(nu.shape)
        for i in range(num_clones-1):
            theta.at[:, i].set(numpyro.sample(f'theta_{i}', numdist.BetaProportion(nu[:, i], 1)))
        clone_props = rho[:-1]
        xi = jnp.sum(theta*clone_props, axis=1)
        numpyro.sample('snv_obs', numdist.Binomial(d, xi), obs=b)

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
                                
