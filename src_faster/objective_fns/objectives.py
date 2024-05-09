import jax.numpy as jnp
from jax import jit,vmap

import jax
from jax.tree_util import Partial as partial
def MPC_decorator(IM_fn,kinematic_model,dt,gamma,fim_method):

    if fim_method == "PFIM":
        @jit
        def MPC_obj(U, radar_state, target_state, J, A):

            # horizon = U.shape[1]
            horizon = target_state.shape[-2]

            radar_states = kinematic_model(U, jnp.tile(radar_state, (U.shape[0], 1, 1)), dt)

            radar_states = radar_states[..., 1:, :3]

            # iterate through time step
            Js = [None]*horizon
            J = jnp.tile(J,(radar_states.shape[0],target_state.shape[0],1,1,1))
            for t in range(1,horizon+1):

                J = IM_fn(radar_state=radar_states[...,t,:],
                          target_state=target_state[...,t-1,:], J=J)
                Js[t-1] = J.mean(axis=1)

                J = J.reshape(radar_states.shape[:1] + target_state.shape[:2] + (6, 6), order="F").mean(axis=1)
                J = jnp.tile(jnp.expand_dims(J,1),(1,target_state.shape[0],1,1,1))

            Jstack = jnp.stack(Js,axis=-3)

            # n_sigma_pts = target_state.shape[0]
            logdets = jnp.linalg.slogdet(Jstack)[1].sum(-2)
            gammas = gamma ** (jnp.arange(horizon))
            multi_FIM_obj = jnp.sum(gammas * logdets, axis=-1) / jnp.sum(gammas)

            return -multi_FIM_obj
    else:
        @jit
        def MPC_obj(U,radar_state,target_state,J,A):

            # horizon = U.shape[1]
            horizon = target_state.shape[-2]

            radar_states = kinematic_model(U,jnp.tile(radar_state,(U.shape[0],1,1)),dt)

            radar_states = radar_states[...,1:,:3]
            J = IM_fn(radar_state=jnp.reshape(radar_states,order="F",newshape=(-1,) + radar_states.shape[-2:]),
                      target_state=jnp.reshape(target_state,order="F",newshape=(-1,) + target_state.shape[-2:]),J=J)

            J  = J.reshape(radar_states.shape[:2] + target_state.shape[:2] + (horizon,3,3),order="F").sum(axis=1)#

            # n_sigma_pts = target_state.shape[0]
            logdets = jnp.linalg.slogdet(J)[1].sum(-2).mean(axis=1)
            gammas = gamma**(jnp.arange(horizon))
            multi_FIM_obj = jnp.sum(gammas*logdets,axis=-1)/jnp.sum(gammas)

            return -multi_FIM_obj

    return MPC_obj

@jit
def collision_penalty(radar_states,target_states,radius):

    # N,horizon,dn= radar_states.shape
    # M,horizon,dm = target_states.shape
    horizon = radar_states.shape[-2]
    radar_positions = radar_states[...,:3]

    target_positions = target_states[...,:3]

    radar_positions=jnp.reshape(radar_positions, order="F", newshape=(-1,) + radar_positions.shape[-2:])
    target_positions =jnp.reshape(target_positions, order="F", newshape=(-1,) + target_positions.shape[-2:])

    d = (radar_positions[:,jnp.newaxis]-target_positions[jnp.newaxis])

    distances = jnp.sqrt(jnp.sum(d**2,-1))

    distances = distances.reshape(radar_states.shape[:2] + target_states.shape[:2] + (horizon, ), order="F")#.sum(axis=1).mean(axis=1)

    coll_obj = (distances < radius)
    # coll_obj = jnp.heaviside(-(distances - radius), 1.0) * jnp.exp(-distances / spread)
    return jnp.sum(coll_obj,axis=[1,3]).mean(axis=1)

@jit
def self_collision_penalty(radar_states,radius):
    # N,horizon,dn = radar_states.shape

    # idx = jnp.arange(N)[:, None] < jnp.arange(N)
    radar_positions = radar_states[...,:3]


    difference = (radar_states[jnp.newaxis] - radar_states[:, jnp.newaxis])
    distances = jnp.sqrt(jnp.sum(difference ** 2, -1))

    coll_obj = (distances < radius).T

    return jnp.sum(jnp.triu(coll_obj,k=1),axis=[-2,-1])


@partial(jit,static_argnames=['dTraj','dN','dC'])
def control_penalty(U_prime,U,V,cov,dTraj,dN,dC):
    cost_control = (U_prime - U).reshape(1, dN, 1, -1) @ jnp.linalg.inv(cov) @ (V).reshape(dTraj, dN, -1, 1)

    return cost_control

@partial(jit,static_argnames=['speed_minimum'])
def speed_penalty(speed,speed_minimum):

    cost_speed =  jnp.sum((jnp.abs(speed) < speed_minimum)*1,0)

    return cost_speed