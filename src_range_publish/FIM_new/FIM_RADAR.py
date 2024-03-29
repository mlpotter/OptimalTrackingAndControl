
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

from src_range_publish.control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator

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
def Single_JU_FIM_Radar(radar_state,target_state,J,A,Q,C):

    N,dn= radar_state.shape
    M,dm = target_state.shape

    radar_positions = radar_state[:,:3]

    target_positions = target_state[:,:3]

    # Qinv = jnp.linalg.inv(Q+jnp.eye(dm*M)*1e-8)
    # # Qinv = jnp.linalg.inv(Q)
    #
    # D11 = A.T @ Qinv @ A
    # D12 = -A.T @ Qinv
    # D21 = D12.T

    d = (target_positions[:,jnp.newaxis,:] - radar_positions[jnp.newaxis,:,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    # jnp.einsum("ijk,ilm->ikm", d, d)
    coef = jnp.sqrt((4/(C*distances**6) + 8/(distances**4)))
    outer_vector = d * coef
    outer_product = jnp.einsum("ijk,ijl->ijkl",outer_vector,outer_vector).sum(axis=0)#(outer_vector.transpose(1,2,0) @ outer_vector.transpose(1,0,2))
    J_standard = jax.scipy.linalg.block_diag(*[jax.scipy.linalg.block_diag(outer_product[m],jnp.zeros((dm-3,dm-3))) for m in range(M)])

    # D22 = J_standard + Qinv

    # J = D22 - D21 @ jnp.linalg.inv(J + D11) @ D12
    # J = D22 - D21 @ jnp.linalg.solve(J+D11,jnp.eye(J.shape[0])) @ D12

    # J = jax.scipy.linalg.block_diag(*[outer_product[m] for m in range(M)])

    J = jnp.linalg.inv(Q+A@jnp.linalg.inv(J)@A.T) + J_standard

    return J


@jit
def JU_RANGE_FIM(radar_state,target_state,J,A,Q,R):

    N,dn= radar_state.shape
    M,dm = target_state.shape

    radar_positions = radar_state[:,:3]

    target_positions = target_state[:,:3]

    d = (target_positions[jnp.newaxis,:,:] - radar_positions[:,jnp.newaxis,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    outer_vector = 2*d/distances

    # outer_product = jnp.einsum("ijk,ijl->ijkl",outer_vector,outer_vector).sum(axis=0) * 1/ (sigmaR**2)#(outer_vector.transpose(1,2,0) @ outer_vector.transpose(1,0,2))

    J_standard = outer_vector.squeeze().T @ jnp.linalg.inv(R) @ outer_vector.squeeze()
    # J_standard = jax.scipy.linalg.block_diag(*[jax.scipy.linalg.block_diag(outer_product[m],jnp.zeros((dm-3,dm-3))) for m in range(M)])
    J_standard = jax.scipy.linalg.block_diag(J_standard,jnp.zeros((dm-3,dm-3)))
    J = jnp.linalg.inv(Q + A@jnp.linalg.inv(J)@A.T) + J_standard
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
