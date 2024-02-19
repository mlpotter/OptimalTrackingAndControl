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


def generate_samples(key, XT, PT, A, Q,
                     Gt,Gr,Pt,lam,rcs,L,c,B,alpha,sigmaV,
                     TN):
    _, subkey = jax.random.split(key)

    M, nx = XT.shape
    N, ny = PT.shape

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

        # measure onward!
        YT = RangeVelocityMeasure(XT_noise.reshape(M, nx), PT).ravel()



        # append state and measurement
        XT_trajectories.append(XT_noise.reshape(-1, ))
        # YT_trajectories.append(YT_noise.reshape(N, )) # if I sum the radar powers
        YT_trajectories.append(YT_noise.reshape(N * M, ))

    return key, jnp.stack(XT_trajectories, axis=0), jnp.stack(YT_trajectories, axis=0)


def optimal_importance_dist_sample(key, Xprev, A, Q):
    """
    Xprev: Nxd matrix of previous samples
    """

    mu = (A @ Xprev.T).T

    key, subkey = random.split(key)

    Xnext = random.multivariate_normal(key=subkey, mean=mu, cov=Q, method='svd')

    return key, Xnext


def weight_update(Wprev, Vnext, s,ynext):
    # Rayleigh
    # Wnext =  Wprev * ss.expon.pdf(ynext.ravel(),scale=Vnext.squeeze()+PN,loc=0).prod(axis=1,keepdims=True)

    # Rice
    # rice_power(Pr=ynext.T,K=K, Pr_avg=Vnext.squeeze()).prod(axis=1,keepdims=True) #jnp.expand_dims(Wnext(Vnext),1)

    # RICE + AWGN
    # Amp, Ma, zeta, s = NoiseParams(Vnext.squeeze(), SCNR, CNR=CNR)
    Wnext = Wprev * ss.ncx2.pdf(ynext.ravel(), df=2, nc=Vnext.squeeze() / (s ** 2), scale=s ** 2).prod(axis=1,
                                                                                                       keepdims=True)

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