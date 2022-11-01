import numpyro
import numpyro.distributions as numdist
import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey

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
    :param counts: (L, 2+K) numpy array - L=length of intersection of SNV positions, 2+K=ref and alt counts for liquid biopsy and num clones (normal is not a clone)
    :param num_clones: integer
    """
    rho = numpyro.sample('rho', numdist.Dirichlet(jnp.ones(num_clones)))
    mu = jnp.log(jnp.sum(cn_profiles*rho, axis=1)) - jnp.log(jnp.mean(jnp.sum(cn_profiles*rho, axis=1)))
    tau = numpyro.sample('tau', numdist.InverseGamma(3, 1))
    with numpyro.plate('data', size=len(data)):
        numpyro.sample('obs', numdist.StudentT(df=4, loc=mu, scale=tau), obs=data)

    xi = jnp.sum(counts[:, 2:]*rho[:-1], axis=1)
    d = jnp.sum(counts[:, :2], axis=1)
    b = counts[:, 1]
    with numpyro.plate('snv', size=len(xi)):
        numpyro.sample('snv_obs', numdist.BinomialLogits(xi, d), obs=b)
