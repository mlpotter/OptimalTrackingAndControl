import jax.numpy as jnp
from jax import jit,vmap
from src_range.control.Sensor_Dynamics import unicycle_kinematics
import jax

def MPC_decorator(IM_fn,method="action"):

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
        def MPC_obj(U,chis,radar_states,target_states,time_step_size,J,
                             A,
                             gamma):
            horizon = U.shape[1]
            M,dm = target_states.shape
            N,dn = radar_states.shape

            _,_,ps_trajectory,chis_trajectory = vmap(unicycle_kinematics,(0,0,0,None))(radar_states,U,chis,time_step_size)

            multi_FIM_obj = 0

            total = 0
            # iterate through time step
            for t in jnp.arange(1,horizon+1):
                # iterate through each FIM corresponding to a target

                J = IM_fn(radar_states=ps_trajectory[:,t],target_states=target_states,J=J)

                _,logdet = jnp.linalg.slogdet(J)
                multi_FIM_obj += gamma**(t-1) * logdet
                total += gamma**(t-1)


                target_states = (A @ target_states.reshape(-1, M*dm).T).T.reshape(M, dm)

            return -multi_FIM_obj/total

    elif method=="Single_FIM_3D_action_MPPI":
        @jit
        def MPC_obj(U,chis,radar_states,target_states,time_step_size,J,
                             A,
                             gamma):
            # horizon = U.shape[1]
            horizon,M,dm = target_states.shape
            N,dn = radar_states.shape

            _,_,ps_trajectory,chis_trajectory = vmap(unicycle_kinematics,(0,0,0,None))(radar_states,U,chis,time_step_size)

            # iterate through time step
            Js = [None]*horizon
            for t in range(1,horizon+1):
                # iterate through each FIM corresponding to a target

                J = IM_fn(radar_states=ps_trajectory[:,t],target_states=target_states[t-1],J=J)
                Js[t-1] = J

            Js = jnp.stack(Js)
            _,logdets = jnp.linalg.slogdet(Js)
            gammas = gamma**(jnp.arange(horizon))
            multi_FIM_obj = jnp.sum(gammas*logdets)/jnp.sum(gammas)

            return -multi_FIM_obj

    elif method =="Single_FIM_Collision":
        @jit
        def MPC_obj(U,chis,radar_states,target_states,time_step_size,J,
                             A,
                             gamma,radius,spread,alpha1,alpha2):
            # horizon = U.shape[1]
            horizon,M,dm = target_states.shape
            N,dn = radar_states.shape

            _,_,ps_trajectory,chis_trajectory = vmap(unicycle_kinematics,(0,0,0,None))(radar_states,U,chis,time_step_size)

            # iterate through time step
            Js = [None]*horizon

            J_coll = 0
            for t in range(1,horizon+1):
                # iterate through each FIM corresponding to a target

                J = IM_fn(radar_states=ps_trajectory[:,t],target_states=target_states[t-1],J=J)
                J_coll = J_coll +  gamma**t * collision_penalty(radar_states=ps_trajectory[:,t],target_states=target_states[t-1],radius=radius,spread=spread)
                Js[t-1] = J

            Js = jnp.stack(Js)
            _,J_info = jnp.linalg.slogdet(Js)
            gammas = gamma**(jnp.arange(horizon))
            J_info = jnp.sum(gammas*J_info)
            MPC_obj = (J_info*alpha1 - J_coll*alpha2)/jnp.sum(gammas)

            return -MPC_obj


    elif method=="Single_FIM_2D_noaction":
        @jit
        def MPC_obj(radar_states, target_states):

            M, dm = target_states.shape
            N, dn = radar_states.shape
            # ps = jnp.expand_dims(ps,1)

            sign, logdet = jnp.linalg.slogdet(IM_fn(radar_states=radar_states,target_states=target_states))

            return -logdet


    return MPC_obj

def collision_penalty(radar_states,target_states,radius,spread):

    N,dn= radar_states.shape
    M,dm = target_states.shape

    radar_states = jnp.concatenate((radar_states,jnp.zeros((N,1))),-1)
    radar_states = radar_states[:,:dm//2]

    target_positions = target_states[:,:dm//2]

    d = (target_positions[jnp.newaxis,:,:] - radar_states[:,jnp.newaxis,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    coll_obj = jnp.exp(-distances / spread)  * (distances < radius)

    return jnp.sum(coll_obj)

def control_penalty(U_prime,U,V,cov,dTraj,dN,dC):
    cost_control = (U_prime - U).reshape(1, dN, 1, -1) @ jnp.linalg.inv(cov) @ (V).reshape(dTraj, dN, -1, 1)

    return cost_control
