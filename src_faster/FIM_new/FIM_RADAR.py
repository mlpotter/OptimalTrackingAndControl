
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
import numpy as np
from scipy.spatial import distance_matrix
from jax.tree_util import Partial as partial

from copy import deepcopy

from control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator

@jit
def Single_FIM_Radar(radar_state,target_state,C,J=None):
    N,dn= radar_state.shape
    M,dm = target_state.shape

    radar_positions = radar_state[:,:3]

    target_positions = target_state[:,:3]

    d = (target_positions[jnp.newaxis,:,:] - radar_positions[:,jnp.newaxis,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))
    
    coef = jnp.sqrt((4/(C*distances**6) + 8/(distances**4)))

    outer_vector = d * coef

    outer_product = (outer_vector.transpose(1,2,0) @ outer_vector.transpose(1,0,2))

    J = jax.scipy.linalg.block_diag(*[outer_product[m] for m in range(M)])
    return J


@jit
def SFIM_parallel(radar_state, target_state, C, J=None):
    # N, T,  dn = radar_state.shape[:-2],target_state.shape[-2:]
    # M, T, dm = target_state.shape[:-2],target_state.shape[-2:]

    radar_ndims = radar_state.ndim
    target_ndims = target_state.ndim

    radar_positions = radar_state[..., :3]

    target_positions = target_state[..., :3]
    d = jnp.expand_dims(target_positions,0) - jnp.expand_dims(radar_positions,1)

    # d = (target_positions[jnp.newaxis, :, :] - radar_positions[:, jnp.newaxis, :])

    distances = jnp.sqrt(jnp.sum(d ** 2, -1, keepdims=True))

    coef = jnp.sqrt((4 / (C * distances ** 6) + 8 / (distances ** 4)))

    outer_vector = d * coef

    outer_product = (outer_vector[...,jnp.newaxis] @ outer_vector[...,jnp.newaxis,:])

    return outer_product

@jit
def SFIM_range(radar_state,target_state,sigmaR,J=None):

    radar_ndims = radar_state.ndim
    target_ndims = target_state.ndim

    radar_positions = radar_state[...,:3]

    target_positions = target_state[...,:3]

    d = jnp.expand_dims(target_positions,0) - jnp.expand_dims(radar_positions,1)

    distances = jnp.sqrt(jnp.sum(d ** 2, -1, keepdims=True))

    outer_vector = 2*d/distances * 1/sigmaR

    # outer_product = jnp.einsum("ijk,ijl->ijkl",outer_vector,outer_vector).sum(axis=0) * 1/ (sigmaR**2)#(outer_vector.transpose(1,2,0) @ outer_vector.transpose(1,0,2))

    outer_product = (outer_vector[...,jnp.newaxis] @ outer_vector[...,jnp.newaxis,:])

    # J_standard = jax.scipy.linalg.block_diag(*[jax.scipy.linalg.block_diag(outer_product[m],jnp.zeros((dm-3,dm-3))) for m in range(M)])
    return outer_product

@jit
def PFIM_parallel(radar_state,target_state,J,A,Q,C):
    radar_state_ravel = jnp.reshape(radar_state, order="F", newshape=(-1,) + radar_state.shape[-1:])
    target_state_ravel  = jnp.reshape(target_state, order="F", newshape=(-1,) + target_state.shape[-1:])

    radar_ndims = radar_state.ndim
    target_ndims = target_state.ndim

    radar_positions = radar_state_ravel[...,:3]

    target_positions = target_state_ravel[...,:3]

    d = jnp.expand_dims(target_positions,0) - jnp.expand_dims(radar_positions,1)

    distances = jnp.sqrt(jnp.sum(d ** 2, -1, keepdims=True))


    coef = jnp.sqrt((4/(C*distances**6) + 8/(distances**4)))
    outer_vector = jnp.concatenate((d,jnp.zeros(d.shape)),axis=-1) * coef

    J_standard = (jnp.expand_dims(outer_vector, -1) @ jnp.expand_dims(outer_vector, -2))#.sum(axis=0)

    J_standard = J_standard.reshape(radar_state.shape[:radar_state.ndim-1] + target_state.shape[:target_state.ndim-1] + (6, 6), order="F").sum(axis=radar_state.ndim-2)

    J = jnp.linalg.inv(Q+A@jnp.linalg.inv(J)@A.T) + J_standard

    return J

@partial(jit,static_argnames=['N',"space"])
def FIM_2D_Visualization(ps,qs,C,N,space):
    ps = ps[:,:2]
    qs = qs[:,:2]

    sensor_and_targets = jnp.vstack((ps,qs))

    x_max,y_max = jnp.min(sensor_and_targets,axis=0)-space
    x_min,y_min = jnp.max(sensor_and_targets,axis=0)+space

    qx,qy = jnp.meshgrid(jnp.linspace(x_min,x_max,N),
                         jnp.linspace(y_min,y_max,N))

    qs = jnp.column_stack((qx.ravel(),qy.ravel()))

    d = (qs[:,jnp.newaxis,:] - ps[jnp.newaxis,:,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    # jnp.einsum("ijk,ilm->ikm", d, d)
    coef = jnp.sqrt((4/(C*distances**6) + 8/(distances**4)))

    outer_vector = d * coef

    J = (outer_vector.transpose(0,2,1) @ outer_vector)

    sign,logdet = jnp.linalg.slogdet(J)

    return qx,qy,logdet.reshape(N,N)





@partial(jit,static_argnames=['N'])
def FIM_Visualization(ps,qs,C,N):

    ps = ps[:,:2];
    qs = qs[:,:2];

    sensor_and_targets = jnp.vstack((ps,qs[:,:3]))[:,:2]

    x_max,y_max = jnp.min(sensor_and_targets,axis=0)-300
    x_min,y_min = jnp.max(sensor_and_targets,axis=0)+300

    qx,qy = jnp.meshgrid(jnp.linspace(x_min,x_max,N),
                         jnp.linspace(y_min,y_max,N))

    qs = jnp.column_stack((qx.ravel(),qy.ravel()))

    d = (qs[:,jnp.newaxis,:] - ps[jnp.newaxis,:,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    coef = jnp.sqrt((4/(C*distances**6) + 8/(distances**4)))

    outer_vector = jnp.expand_dims(d * coef,-1)

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    outer_product =  outer_vector @ outer_vector.transpose(0,1,3,2)

    J = jnp.sum(outer_product, axis=1)

    sign,logdet = jnp.linalg.slogdet(jnp.linalg.inv(J)+jnp.tile(jnp.eye(J.shape[-1]),(N**2,1,1))*1e-8)

    logdet =  -logdet

    return qx,qy,logdet.reshape(N,N)
