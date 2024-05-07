# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

import jax

import jax.numpy as jnp
from jax import vmap
import functools
import jax
from jax import jit
from jax.tree_util import Partial as partial

import numpy as np

import scipy.stats as ss
from tqdm import tqdm

from jax import random
from src_range_final.tracking.Measurement_Models import RangeVelocityMeasure


def generate_trajectories(key, XT, A, Q,
                     Gt,Gr,Pt,lam,rcs,L,c,fc,sigmaW,sigmaV,
                     TN):
    _, subkey = jax.random.split(key)

    M, nx = XT.shape

    XT = XT.reshape(-1, 1)

    XT_trajectories = []  # np.zeros((M*4,NT))
    YT_trajectories = []

    key, subkey = random.split(key)
    QK = jax.random.multivariate_normal(subkey, mean=jnp.zeros(Q.shape[0], ), cov=Q, shape=(TN,), method="svd")

    key, subkey = random.split(key)

    for k in tqdm(range(TN)):
        # move forward!
        XT = (A @ XT).ravel()
        XT_noise = XT + QK[k, :]

        # append state and measurement
        XT_trajectories.append(XT_noise.reshape(-1, ))

    return key, jnp.stack(XT_trajectories, axis=0)


def optimal_importance_dist_sample(key, Xprev, A, Q):
    """
    Xprev: Nxd matrix of previous samples
    """

    mu = (A @ Xprev.T).T

    key, subkey = random.split(key)

    Xnext = random.multivariate_normal(key=subkey, mean=mu, cov=Q, method='svd')

    return key, Xnext

def propogate_optimal_importance_dist_sample(key,Xprev,A,Q,horizon):
    X_rollout = [None]*horizon

    for t in range(1,horizon+1):
        Xprev = A @ Xprev.T
        X_rollout[t-1] = Xprev.T

        key, subkey = random.split(key)

        Xprev = random.multivariate_normal(key=subkey, mean=Xprev.T, cov=Q, method='svd')

    return key, jnp.stack(X_rollout,axis=1)

@partial(jax.jit, static_argnames=['M','N','dm','dn'])
def weight_update(Wprev, Vnext,ynext,C,sigmaV,M,N,dm,dn):

    ynext = ynext.reshape(M*N,dm//2 + 1)
    velocity_measures = Vnext.reshape(-1, M * N, (dm // 2 + 1))[:, :, 1:]
    range_measures = Vnext.reshape(-1,M * N, (dm // 2 + 1))[:, :, :1]

    velocity_pdf = vmap(lambda velocities: jax.scipy.stats.multivariate_normal.pdf(ynext[:,1:].ravel(),mean=velocities.ravel(),cov=sigmaV**2),in_axes=(0,))

    velocity_pdf = velocity_pdf(velocity_measures) #jax.scipy.stats.multivariate_normal.pdf(ynext[:,1:],mean=velocity_measures,
                                        #cov=jnp.eye((dm // 2)) * sigmaV**2).prod(axis=-1)


    range_pdf = vmap(lambda ranges: jax.scipy.stats.multivariate_normal.pdf(ynext[:,:1].ravel(),mean=ranges.ravel(),cov=jnp.diag(ranges.ravel()**4)*C),in_axes=(0,))

    range_pdf = range_pdf(range_measures)


    Wnext = jnp.reshape(velocity_pdf*range_pdf,(-1,1)) * Wprev

    Wnext = Wnext / np.sum(Wnext)

    #     print("Number of NaNs: ",jnp.isnan(Wnext).sum())

    return Wnext


def effective_samples(W):
    return 1 / (np.sum(W ** 2))


def weight_resample(Xnext, Wnext):
    NP = Wnext.shape[0]
    idx = np.random.choice(NP, size=(NP,), p=Wnext.ravel())
    Xnext = Xnext[idx]
    Wnext = np.ones((NP, 1)) * 1 / NP

    return Xnext, Wnext