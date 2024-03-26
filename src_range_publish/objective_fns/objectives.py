import jax.numpy as jnp
from jax import jit,vmap

import jax
from jax.tree_util import Partial as partial
def MPC_decorator(IM_fn,kinematic_model,dt,gamma,method="action"):

    # the lower this value, the better!

    if method=="Single_FIM_3D_action":
        @jit
        def MPC_obj(U,radar_state,target_state,J,A):

            horizon = U.shape[1]
            M,dm = target_state.shape
            N,dn = radar_state.shape

            radar_states = kinematic_model(U,radar_state,dt)

            multi_FIM_obj = 0

            total = 0
            # iterate through time step
            for t in jnp.arange(1,horizon+1):
                # iterate through each FIM corresponding to a target

                J = IM_fn(radar_state=radar_states[:,t],target_state=target_state,J=J)

                _,logdet = jnp.linalg.slogdet(J)
                multi_FIM_obj += gamma**(t-1) * logdet
                total += gamma**(t-1)


                target_state = (A @ target_state.reshape(-1, M*dm).T).T.reshape(M, dm)

            return -multi_FIM_obj/total

    elif method=="Single_FIM_3D_action_MPPI":
        @jit
        def MPC_obj(U,radar_state,target_state,J,A):

            # horizon = U.shape[1]
            M,horizon,dm = target_state.shape
            N,dn = radar_state.shape

            radar_states = kinematic_model(U,radar_state,dt)

            # iterate through time step
            Js = [None]*horizon
            for t in range(1,horizon+1):
                # iterate through each FIM corresponding to a target

                J = IM_fn(radar_state=radar_states[:,t,:3],target_state=target_state[:,t-1],J=J)
                Js[t-1] = J

            Js = jnp.stack(Js)
            _,logdets = jnp.linalg.slogdet(Js)
            gammas = gamma**(jnp.arange(horizon))
            multi_FIM_obj = jnp.sum(gammas*logdets)/jnp.sum(gammas)

            return -multi_FIM_obj

    elif method=="Single_FIM_2D_noaction":
        @jit
        def MPC_obj(radar_states, target_states):

            M, dm = target_states.shape
            N, dn = radar_states.shape
            # ps = jnp.expand_dims(ps,1)

            sign, logdet = jnp.linalg.slogdet(IM_fn(radar_states=radar_states,target_states=target_states))

            return -logdet


    return MPC_obj

@jit
def collision_penalty(radar_states,target_states,radius):

    # N,horizon,dn= radar_states.shape
    # M,horizon,dm = target_states.shape

    radar_positions = radar_states[...,:3]

    target_positions = target_states[...,:3]

    d = (radar_positions[:,jnp.newaxis]-target_positions[jnp.newaxis])

    distances = jnp.sqrt(jnp.sum(d**2,-1))

    coll_obj = (distances < radius)
    # coll_obj = jnp.heaviside(-(distances - radius), 1.0) * jnp.exp(-distances / spread)
    return jnp.sum(coll_obj,axis=[0,1])

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