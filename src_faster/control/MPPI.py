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

from sklearn.covariance import OAS,ledoit_wolf,oas

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

def MPPI_control(
                radar_state,U,cov,key, # radar state
                 A,J,control_constraints, # objective parameters
                 kinematic_model,ckf, # kinematic model and cubature kalman filter
                 MPPI_kinematics,MPPI_scores,weight_fn, # MPPI need parameters
                 collision_penalty,self_collision_penalty_vmap, # extra penalty functions
                 args # misc things
                 ):

    # number of radars x horizon x 2
    U_prime = deepcopy(U)
    cov_prime = deepcopy(cov)

    # the cubature kalman filter points propogated over horizon. Horizon x # Sigma Points x (Number of targets * dim of target)
    target_states_ckf = ckf.predict_propogate(ckf.x, ckf.P, args.horizon, dt=args.dt_control, fx_args=(args.M_target,))
    # target_states_ckf = np.swapaxes(target_states_ckf.mean(axis=1).reshape(args.horizon, M_target, dm), 1, 0)


    # # Sigma Points x Number of targets x horizon x dm
    target_states_ckf = np.moveaxis(target_states_ckf.reshape(args.horizon, args.dm * args.M_target * 2, args.M_target, args.dm), source=0,
                                    destination=-2)

    for mppi_iter in range(args.MPPI_iterations):
        key, subkey = jax.random.split(key)

        try:
            E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime,
                                               shape=(args.num_traj,))  # ,method="svd")
        except:
            E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime,
                                               shape=(args.num_traj,), method="svd")

        # simulate the model with the trajectory noise samples
        # number of traj x number of radars x horizon x 2
        V = U_prime + E.reshape(args.num_traj, args.N_radar, args.horizon, 2)
        # mppi_sample_end = time()

        # number of radars x horizon+1 x dn
        # number of traj x number of radars x horizon+1 x dn
        radar_states, radar_states_MPPI = MPPI_kinematics(U_nominal=U_prime,
                                               U_MPPI=V, radar_state=radar_state)

        # GET MPC OBJECTIVE
        # mppi_score_start = time()
        # Score all the rollouts
        cost_trajectory = MPPI_scores(V, radar_state, target_states_ckf,
                                      J, A)

        cost_collision_r2t = collision_penalty(radar_states_MPPI[..., 1:args.horizon + 1, :], target_states_ckf,
                                               args.R2T)

        cost_collision_r2t = jnp.sum((cost_collision_r2t * args.gamma ** (jnp.arange(args.horizon))) / jnp.sum(
            args.gamma ** jnp.arange(args.horizon)), axis=-1)

        cost_collision_r2r = self_collision_penalty_vmap(radar_states_MPPI[..., 1:args.horizon + 1, :], args.R2R)
        cost_collision_r2r = jnp.sum((cost_collision_r2r * args.gamma ** (jnp.arange(args.horizon))) / jnp.sum(
            args.gamma ** jnp.arange(args.horizon)), axis=-1)

        cost_MPPI = args.alpha1 * cost_trajectory + args.alpha2 * cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature

        weights = weight_fn(cost_MPPI)

        if jnp.isnan(cost_MPPI).any():
            print("BREAK!")
            break

        if (mppi_iter < (args.MPPI_iterations - 1)):  # and (jnp.sum(cost_MPPI*weights) < best_cost):

            # number of radars x horizon x 2
            U_prime = U_prime + jnp.sum(
                weights.reshape(args.num_traj, 1, 1, 1) * E.reshape(args.num_traj, args.N_radar, args.horizon, 2),
                axis=0)

            # this is only working right now for entropy based weighting
            lw_cov, shrinkage = ledoit_wolf(X=E[weights != 0], assume_centered=True)

            cov_prime = jnp.array(lw_cov)
            if mppi_iter == 0:
                # print("Oracle Approx Shrinkage: ",np.round(shrinkage,5))
                pass

    if jnp.isnan(cost_MPPI).any():
        raise ValueError("Cost is NaN")

    weights = weight_fn(cost_MPPI)

    mean_shift = (U_prime - U)

    E_prime = E + mean_shift.ravel()

    U += jnp.sum(weights.reshape(-1, 1, 1, 1) * E_prime.reshape(args.num_traj, args.N_radar, args.horizon, 2), axis=0)

    U = jnp.stack((jnp.clip(U[:, :, 0], control_constraints[0, 0], control_constraints[1, 0]),
                   jnp.clip(U[:, :, 1], control_constraints[0, 1], control_constraints[1, 1])), axis=-1)


    # generate radar states at measurement frequency
    # number of radar x steps of update freq control x dn
    radar_states = kinematic_model(np.repeat(U, args.update_freq_control, axis=1)[:, :args.update_freq_control, :],
                                   radar_state, args.dt_ckf)


    U = jnp.roll(U, -1, axis=1)

    return U,(radar_states,radar_states_MPPI),(cost_MPPI,cost_trajectory,cost_collision_r2t,cost_collision_r2r),key


def MPPI_visualize(MPPI_trajectories,nominal_trajectory):
    # J_eval = Multi_FIM_Logdet(U, chis, ps, qs, dts=dts, J=J, A=A, Q=Q, W=W, **key_args)
    fig,axes = plt.subplots(1,1)
    num_traj,N,time_steps,d = MPPI_trajectories.shape
    for n in range(N):
        axes.plot(MPPI_trajectories[:, n, :, 0].T, MPPI_trajectories[:, n, :, 1].T,'b-')

    axes.plot(nominal_trajectory[:, :, 0].T, nominal_trajectory[:, :, 1].T, 'r-')
    axes.axis("equal")
    plt.show()
