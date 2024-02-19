import numpy as np
from scipy.spatial import distance_matrix
from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax import vmap

from src_range.control.Sensor_Dynamics import state_multiple_update

from jaxopt import ScipyBoundedMinimize
import matplotlib.pyplot as plt

import imageio


# from src.Measurement import RadarEqnMeasure,ExponentialDecayMeasure
# from jax import jacfwd
# l = jacfwd(RadarEqnMeasure)(qs,ps,Pt,Gt,Gr,L,lam,rcs)
@jit
def MPPI(U_nominal,chis_nominal,U_ptb,ps,time_step_sizes,limits):

    N,T,dc = U_nominal.shape

    # U is Number of sensors x Control Inputs x T
    U_upper = jnp.ones((1,1,T, dc)) * jnp.reshape(limits[0],(1,1,1,dc))
    U_lower = jnp.ones((1,1,T, dc)) * jnp.reshape(limits[1],(1,1,1,dc))


    # U_velocity = jax.random.uniform(key, shape=(num_traj,N, 1, time_steps), minval=limits[1][0], maxval=limits[0][0])
    # U_angular_velocity = jax.random.uniform(key, shape=(num_traj,N, 1, time_steps), minval=limits[1][1],
    #                                         maxval=limits[0][1])
    # U_ptb = jnp.concatenate((U_velocity,U_angular_velocity),axis=2)
    U_MPPI = jnp.clip(jnp.expand_dims(U_nominal,0) + U_ptb,U_lower,U_upper)

    ps_expanded = jnp.expand_dims(ps, 1)

    kinematic_model = vmap(state_multiple_update, (0, 0, 0, 0))

    _,_,nominal_trajectory,nominal_chis = kinematic_model(ps_expanded,U_nominal,chis_nominal,time_step_sizes)

    MPPI_paths = vmap(kinematic_model,(None,0,None,None))

    _, _, ps_trajectory, chis_trajectory = MPPI_paths(ps_expanded, U_MPPI, chis_nominal, time_step_sizes)

    # ps_unexpanded = jnp.squeeze(ps_forward, 1)


    # J_eval = Multi_FIM_Logdet(U, chis, ps, qs, time_step_sizes=time_step_sizes, J=J, A=A, Q=Q, W=W, **key_args)

    return U_MPPI,ps_trajectory,chis_trajectory,U_nominal,nominal_trajectory,nominal_chis

def MPPI_ptb(stds,N, time_steps, num_traj, key):
    v_min,v_max = stds[0]
    av_min,av_max = stds[1]
    # U_velocity = jax.random.uniform(key, shape=(num_traj, N, time_steps,1), minval=v_min, maxval=v_max)
    # U_angular_velocity = jax.random.uniform(key, shape=(num_traj, N, time_steps,1), minval=av_min,
    #                                         maxval=av_max)

    U_velocity = jax.random.beta(key,.5,.5,shape=(num_traj, N, time_steps,1)) * (v_max - v_min) + v_min
    U_angular_velocity = jax.random.beta(key, 0.5,0.5,shape=(num_traj, N, time_steps,1)) * (v_max - v_min) + v_min
    U_ptb = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)

    # U_ptb = jax.random.normal(key, shape=(num_traj, N, 2, time_steps)) * stds.reshape(1, 1, 2, 1)

    return U_ptb

# @jit
@partial(jit,static_argnames=['score_fn'])
def MPPI_scores(score_fn,ps,qs,U_MPPI,chis,time_step_sizes,A, Q, Js,paretos,Pt, Gt, Gr, L, lam, rcs,s,gamma):
    # the lower the value, the better
    score_fn_partial = partial(score_fn,chis=chis, ps=ps, qs=qs, time_step_sizes=time_step_sizes,
                                            A=A,Q=Q,Js=Js,paretos=paretos,
                                            Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,s=s,
                                            gamma=gamma)
    MPPI_score_fn = vmap(score_fn_partial)

    scores = MPPI_score_fn(U_MPPI)

    return scores


def MPPI_visualize(MPPI_trajectories,nominal_trajectory):
    # J_eval = Multi_FIM_Logdet(U, chis, ps, qs, time_step_sizes=time_step_sizes, J=J, A=A, Q=Q, W=W, **key_args)
    fig,axes = plt.subplots(1,1)
    num_traj,N,time_steps,d = MPPI_trajectories.shape
    for n in range(N):
        axes.plot(MPPI_trajectories[:, n, :, 0].T, MPPI_trajectories[:, n, :, 1].T,'b-')

    axes.plot(nominal_trajectory[:, :, 0].T, nominal_trajectory[:, :, 1].T, 'r-')
    axes.axis("equal")
    plt.show()
