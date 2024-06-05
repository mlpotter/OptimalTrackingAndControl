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


@jit
def RangeVelocityMeasure(qs, ps):
    M, dm = qs.shape
    N, dn = ps.shape
    # ps = jnp.concatenate((ps, jnp.array([4, 4, -4, 8]).reshape(N, 1)), -1)  # jnp.zeros((N,1))),-1)
    ps = jnp.concatenate((ps,jnp.zeros((N,1))),-1)
    ps = ps[:,:dm//2]

    vs = jnp.tile(qs[:,dm//2:],(N,1,1))
    qs = qs[:,:dm//2]

    differences = (ps[:, jnp.newaxis] - qs[jnp.newaxis, :])
    ranges = 2*jnp.sqrt(jnp.sum((differences ** 2), -1,keepdims=True))  # + 1e-8


    measure = jnp.concatenate((ranges,vs),axis=-1)

    measure = measure.reshape(N * M, -1)

    return measure


def generate_range_samples(key, XT, PT, A, Q,
                     Gt,Gr,Pt,lam,rcs,L,c,fc,sigmaW,sigmaV,
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
    # noise for the velocities
    RV = jax.random.multivariate_normal(subkey,mean=jnp.zeros(nx//2*M*N,),cov=jnp.eye(nx//2 * M*N)*sigmaV**2,shape=(TN,))

    # coef of noise for RANGE measurements
    K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L
    C = c**2 * sigmaW**2 / (fc**2 * 8 * jnp.pi**2) * 1/K

    key, subkey = random.split(key)

    for k in tqdm(range(TN)):
        # move forward!
        XT = (A @ XT).ravel()
        XT_noise = XT + QK[k, :]

        # measure onward!
        YT = RangeVelocityMeasure(XT_noise.reshape(M, nx), PT).ravel()

        # noise for the ranges
        key, subkey = random.split(key)
        range_measures = YT.reshape(M*N,(nx//2 + 1))[:,:1].ravel()
        rr = jax.random.multivariate_normal(subkey, mean=jnp.zeros(M*N,),cov=C*jnp.diag(range_measures**4), shape=())
        rv = RV[k,:].reshape(M*N,nx//2)
        rr = rr.reshape(M*N,1)
        YT_noise = YT.reshape(M*N,(nx//2+1)) + jnp.concatenate((rr,rv),axis=-1)
        YT_noise = YT_noise.ravel()

        # append state and measurement
        XT_trajectories.append(XT_noise.reshape(-1, ))
        # YT_trajectories.append(YT_noise.reshape(N, )) # if I sum the radar powers
        YT_trajectories.append(YT_noise.reshape(N * M, nx//2 + 1))

    return key, jnp.stack(XT_trajectories, axis=0), jnp.stack(YT_trajectories, axis=0)


def optimal_importance_dist_sample(key, Xprev, A, Q):
    """
    Xprev: Nxd matrix of previous samples
    """

    mu = (A @ Xprev.T).T

    key, subkey = random.split(key)

    Xnext = random.multivariate_normal(key=subkey, mean=mu, cov=Q, method='svd')

    return key, Xnext



def weight_update(Wprev, Vnext,ynext,Pt,Gt,Gr,lam,rcs,L,c,fc,sigmaW,sigmaV,M,N,dm,dn):

    ynext = ynext.reshape(M*N,dm//2 + 1)
    velocity_measures = Vnext.reshape(-1, M * N, (dm // 2 + 1))[:, :, 1:]
    range_measures = Vnext.reshape(-1,M * N, (dm // 2 + 1))[:, :, :1]

    velocity_pdf = vmap(lambda velocities: jax.scipy.stats.multivariate_normal.pdf(ynext[:,1:].ravel(),mean=velocities.ravel(),cov=sigmaV**2),in_axes=(0,))

    velocity_pdf = velocity_pdf(velocity_measures) #jax.scipy.stats.multivariate_normal.pdf(ynext[:,1:],mean=velocity_measures,
                                        #cov=jnp.eye((dm // 2)) * sigmaV**2).prod(axis=-1)

    # coef of noise for RANGE measurements
    K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L
    C = c**2 * sigmaW**2 / (fc**2 * 8 * jnp.pi**2) * 1/K

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