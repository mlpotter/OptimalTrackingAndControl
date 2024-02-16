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
def RadarEqnMeasure(qs, ps,Gt,Gr,Pt,lam,rcs,L):
    N = ps.shape[0]
    M, nx = qs.shape

    ps = jnp.concatenate((ps, jnp.array([4, 4, -4, 8]).reshape(N, 1)), -1)  # jnp.zeros((N,1))),-1)
    qs = qs[:, :3]

    differences = (ps[:, jnp.newaxis] - qs[jnp.newaxis, :])
    distances = jnp.sqrt(jnp.sum((differences ** 2), -1))  # + 1e-8

    coef = Gt * Gr * Pt * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3

    v = (coef * (distances) ** (-2)).reshape(N * M, -1)

    return v


def generate_samples(key, XT, PT, A, Q,
                     Gt,Gr,Pt,lam,rcs,L,s,
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
        YT = RadarEqnMeasure(XT_noise.reshape(M, nx), PT,Gt,Gr,Pt,lam,rcs,L).ravel()

        # sigma,s = SNR2PARAM(SNR,YT)

        # RICE
        # b,scale,loc = SCIPY2RICE(sigma,s)
        # YT_noise = ss.rice.rvs(b=b,scale=scale,loc=loc)**2

        # RAYLEIGH
        # YT_noise = ss.expon.rvs(scale=PN+YT,loc=0)

        #
        # Amp, Ma, zeta, s =  NoiseParams(YT, SCNR, CNR=CNR)

        YT_noise = ss.ncx2.rvs(df=2, nc=YT / (s ** 2), scale=s ** 2)
        # Z = rv.rvs((N,))

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