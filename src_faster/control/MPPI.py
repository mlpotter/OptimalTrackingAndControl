import numpy as np
from scipy.spatial import distance_matrix
from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import Partial as partial
from jax import vmap

from control.Sensor_Dynamics import unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator,UNI_SI_U_LIM,UNI_SI_U_LIM,UNI_DI_U_LIM

from jaxopt import ScipyBoundedMinimize
import matplotlib.pyplot as plt

import imageio
from copy import deepcopy


# from src.Measurement import RadarEqnMeasure,ExponentialDecayMeasure
# from jax import jacfwd
# l = jacfwd(RadarEqnMeasure)(qs,ps,Pt,Gt,Gr,L,lam,rcs)
def MPPI_wrapper(kinematic_model,dt):

    MPPI_paths_vmap = vmap(kinematic_model, (0, None, None))
    @jit
    def MPPI(U_nominal,U_MPPI,radar_state):

        Ntraj,N,T,dc = U_MPPI.shape
        _,dn = radar_state.shape
        # U is Number of sensors x Control Inputs x T


        radar_position = radar_state[:,:3]

        # U_velocity = jax.random.uniform(key, shape=(num_traj,N, 1, time_steps), minval=limits[1][0], maxval=limits[0][0])
        # U_angular_velocity = jax.random.uniform(key, shape=(num_traj,N, 1, time_steps), minval=limits[1][1],
        #                                         maxval=limits[0][1])
        # U_ptb = jnp.concatenate((U_velocity,U_angular_velocity),axis=2)

        # radar_state_expanded = jnp.expand_dims(radar_state, 1)

        radar_states = kinematic_model(U_nominal,radar_state,dt)
        radar_states_MPPI = MPPI_paths_vmap(U_MPPI, radar_state, dt)

        # ps_unexpanded = jnp.squeeze(ps_forward, 1)

        # J_eval = Multi_FIM_Logdet(U, chis, ps, qs, dts=dts, J=J, A=A, Q=Q, W=W, **key_args)

        return radar_states,radar_states_MPPI

    return MPPI


@jit
def MPPI_CMA(U_MPPI,chis_nominal,ps,dt,limits):

    _,N,T,dc = U_MPPI.shape

    # U is Number of sensors x Control Inputs x T
    U_upper = jnp.ones((1,1,T, dc)) * jnp.reshape(limits[0],(1,1,1,dc))
    U_lower = jnp.ones((1,1,T, dc)) * jnp.reshape(limits[1],(1,1,1,dc))


    # U_velocity = jax.random.uniform(key, shape=(num_traj,N, 1, time_steps), minval=limits[1][0], maxval=limits[0][0])
    # U_angular_velocity = jax.random.uniform(key, shape=(num_traj,N, 1, time_steps), minval=limits[1][1],
    #                                         maxval=limits[0][1])
    # U_ptb = jnp.concatenate((U_velocity,U_angular_velocity),axis=2)
    ps_expanded = jnp.expand_dims(ps, 1)

    kinematic_model = vmap(unicycle_kinematics, (0, 0, 0, None))


    MPPI_paths = vmap(kinematic_model,(None,0,None,None))

    _, _, ps_trajectory, chis_trajectory = MPPI_paths(ps_expanded, U_MPPI, chis_nominal, dt)

    # ps_unexpanded = jnp.squeeze(ps_forward, 1)


    # J_eval = Multi_FIM_Logdet(U, chis, ps, qs, dts=dts, J=J, A=A, Q=Q, W=W, **key_args)

    return ps_trajectory,chis_trajectory



def MPPI_ptb(stds,N, time_steps, num_traj, key,method="beta"):
    v_min,v_max = stds[0]
    av_min,av_max = stds[1]
    # U_velocity = jax.random.uniform(key, shape=(num_traj, N, time_steps,1), minval=v_min, maxval=v_max)
    # U_angular_velocity = jax.random.uniform(key, shape=(num_traj, N, time_steps,1), minval=av_min,
    #                                         maxval=av_max)

    if method == "beta":
        U_velocity = jax.random.beta(key,.5,.5,shape=(num_traj, N, time_steps,1)) * (v_max - v_min) + v_min
        U_angular_velocity = jax.random.beta(key, 0.5,0.5,shape=(num_traj, N, time_steps,1)) * (av_max - av_min) + av_min

    elif method=="uniform":
        U_velocity = jax.random.uniform(key,shape=(num_traj, N, time_steps,1)) * (v_max - v_min) + v_min
        U_angular_velocity = jax.random.uniform(key,shape=(num_traj, N, time_steps,1)) * (av_max - av_min) + av_min

    elif method=='normal_biased':
        U_velocity = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * v_max + 1
        U_angular_velocity = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * av_max

    elif method=='normal':
        U_velocity = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * v_max
        U_angular_velocity = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * av_max

    elif method=='mixture':
        p = jnp.array([0.5,0.5])
        select = jax.random.choice(key,a=2,shape=(num_traj,1,1,1),p=p)
        U_velocity_normal = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * v_max
        U_angular_velocity_normal = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * av_max

        U_velocity_beta = jax.random.beta(key,.5,.5,shape=(num_traj, N, time_steps,1)) * (v_max - v_min) + v_min
        U_angular_velocity_beta = jax.random.beta(key, 0.5,0.5,shape=(num_traj, N, time_steps,1)) * (av_max - av_min) + av_min

        U_velocity = jnp.where(select == 1, U_velocity_normal, U_velocity_beta)
        U_angular_velocity = jnp.where(select == 1, U_angular_velocity_normal, U_angular_velocity_beta)

    elif method=='mixture_biased':
        p = jnp.array([0.5,0.5])
        select = jax.random.choice(key,a=2,shape=(num_traj,1,1,1),p=p)
        U_velocity_normal = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * v_max + 1
        U_angular_velocity_normal = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * av_max

        U_velocity_beta = jax.random.beta(key,.5,.5,shape=(num_traj, N, time_steps,1)) * (v_max - v_min) + v_min
        U_angular_velocity_beta = jax.random.beta(key, 0.5,0.5,shape=(num_traj, N, time_steps,1)) * (av_max - av_min) + av_min

        U_velocity = jnp.where(select == 1, U_velocity_normal, U_velocity_beta)
        U_angular_velocity = jnp.where(select == 1, U_angular_velocity_normal, U_angular_velocity_beta)

    U_ptb = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)

    # U_ptb = jax.random.normal(key, shape=(num_traj, N, 2, time_steps)) * stds.reshape(1, 1, 2, 1)

    return U_ptb

def MPPI_ptb_CMA(mu,cov,N, num_traj, key):

    U_velocity = jax.random.multivariate_normal(key,mean=mu,cov=cov,shape=(num_traj, ))
    U_angular_velocity = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * av_max + 0.2


    U_ptb = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)

    # U_ptb = jax.random.normal(key, shape=(num_traj, N, 2, time_steps)) * stds.reshape(1, 1, 2, 1)

    return U_ptb

def weighting(method="CE"):

    if method == "CE":
        def weight_fn(costs,elite_threshold=0.8):
            num_traj = costs.shape[0]

            zeta = jnp.round(num_traj * (1-elite_threshold))
            score_zeta = jnp.quantile(costs,1-elite_threshold)

            weight = 1/zeta * (costs <= score_zeta)
            return weight/jnp.sum(weight,0)


    elif method == "information":
        def weight_fn(costs,temperature):

            weight = jax.nn.softmax(-1/temperature * (costs-jnp.min(costs,axis=0)),axis=0)
            return weight

    return weight_fn


def MPPI_scores_wrapper(score_fn,method="single"):

    @jit
    def MPPI_scores(radar_state,target_state,U_MPPI,A,J):
        # the lower the value, the better
        score_fn_partial = partial(score_fn, radar_state=radar_state, target_state=target_state,
                                                A=A,J=J)

        MPPI_score_fn = vmap(score_fn_partial)


        costs = MPPI_score_fn(U_MPPI)

        return costs


    return MPPI_scores



def MPPI_visualize(MPPI_trajectories,nominal_trajectory):
    # J_eval = Multi_FIM_Logdet(U, chis, ps, qs, dts=dts, J=J, A=A, Q=Q, W=W, **key_args)
    fig,axes = plt.subplots(1,1)
    num_traj,N,time_steps,d = MPPI_trajectories.shape
    for n in range(N):
        axes.plot(MPPI_trajectories[:, n, :, 0].T, MPPI_trajectories[:, n, :, 1].T,'b-')

    axes.plot(nominal_trajectory[:, :, 0].T, nominal_trajectory[:, :, 1].T, 'r-')
    axes.axis("equal")
    plt.show()
