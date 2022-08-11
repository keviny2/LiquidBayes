import os
import numpyro
import numpyro.distributions as numdist
import numpy as np
import jax.numpy as jnp


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
