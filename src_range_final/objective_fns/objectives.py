import jax.numpy as jnp
from jax import jit,vmap
# from src_range.control.Sensor_Dynamics import unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator

import jax
from jax.tree_util import Partial as partial
def MPC_decorator(IM_fn,kinematic_model,time_step_size,gamma,method="action"):

    # the lower this value, the better!


    if method=="Multiple_FIM_2D_action":
        @jit
        def MPC_obj(U,chis,radar_states,target_states,time_step_size,Js,paretos,
                             A,
                             gamma):
            horizon = U.shape[1]
            M,dm = target_states.shape
            N,dn = radar_states.shape

            _,_,ps_trajectory,chis_trajectory = vmap(unicycle_kinematics,(0,0,0,None))(radar_states,U,chis,time_step_size)

            multi_FIM_obj = 0

            IM_parallel = vmap(IM_fn,in_axes=(None,0,0))

            # iterate through time step
            for t in jnp.arange(1,horizon+1):
                # iterate through each FIM corresponding to a target

                Js = IM_parallel(ps_trajectory[:,t],target_states,Js)

                _,logdets = jnp.linalg.slogdet(Js)
                multi_FIM_obj += jnp.sum(gamma**(t-1) * paretos * logdets.ravel())


                target_states = (A @ target_states.reshape(-1, dm).T).T.reshape(M, dm)

            return -multi_FIM_obj

    elif method=="Single_FIM_3D_action":
        @jit
        def MPC_obj(U,radar_state,target_state,J,A):

            horizon = U.shape[1]
            M,dm = target_state.shape
            N,dn = radar_state.shape

            radar_states = vmap(kinematic_model,(0,0,None))(U,jnp.expand_dims(radar_state,1),time_step_size)

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
            horizon,M,dm = target_state.shape
            N,dn = radar_state.shape

            radar_states = vmap(kinematic_model,(0,0,None))(U,jnp.expand_dims(radar_state,1),time_step_size)

            # iterate through time step
            Js = [None]*horizon
            for t in range(1,horizon+1):
                # iterate through each FIM corresponding to a target

                J = IM_fn(radar_state=radar_states[:,t,:2],target_state=target_state[t-1],J=J)
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
def collision_penalty(radar_states,target_states,radius,spread):

    N,dn= radar_states.shape
    M,dm = target_states.shape

    radar_states = jnp.concatenate((radar_states,jnp.zeros((N,1))),-1)
    radar_states = radar_states[:,:dm//2]

    target_positions = target_states[:,:dm//2]

    d = (target_positions[jnp.newaxis,:,:] - radar_states[:,jnp.newaxis,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    coll_obj = jnp.exp(-distances / spread) * (distances < radius)
    # coll_obj = jnp.heaviside(-(distances - radius), 1.0) * jnp.exp(-distances / spread)
    return jnp.sum(coll_obj)

@jit
def self_collision_penalty(radar_states,radius,spread):
    N,dn = radar_states.shape
    # idx = jnp.arange(N)[:, None] < jnp.arange(N)
    difference = (radar_states[jnp.newaxis, :, :] - radar_states[:, jnp.newaxis, :])
    distances = jnp.sqrt(jnp.sum(difference ** 2, -1))

    coll_obj = jnp.exp(-distances / spread) * (distances < radius)

    return jnp.sum(jnp.triu(coll_obj,k=1))


@partial(jit,static_argnames=['dTraj','dN','dC'])
def control_penalty(U_prime,U,V,cov,dTraj,dN,dC):
    cost_control = (U_prime - U).reshape(1, dN, 1, -1) @ jnp.linalg.inv(cov) @ (V).reshape(dTraj, dN, -1, 1)

    return cost_control

@partial(jit,static_argnames=['speed_minimum'])
def speed_penalty(speed,speed_minimum):

    cost_speed =  jnp.sum((jnp.abs(speed) < speed_minimum)*1,0)

    return cost_speed