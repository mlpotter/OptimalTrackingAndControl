# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

import jax

import jax.numpy as jnp
from jax import vmap
import functools
import jax
from jax import jit

import numpy as np

import scipy.stats as ss
from tqdm import tqdm

from jax import random


@jit
def RangeVelocityMeasure(qs, ps):
    M, dm = qs.shape
    N, dn = ps.shape
    # ps = jnp.concatenate((ps, jnp.array([4, 4, -4, 8]).reshape(N, 1)), -1)  # jnp.zeros((N,1))),-1)
    vs = jnp.tile(qs[:,dm//2:],(N,1,1))
    qs = qs[:,:dm//2]

    differences = (ps[:, jnp.newaxis] - qs[jnp.newaxis, :])
    ranges = jnp.sqrt(jnp.sum((differences ** 2), -1,keepdims=True))  # + 1e-8


    measure = jnp.concatenate((ranges,vs),axis=-1)

    measure = measure.reshape(N * M, -1)

    return measure