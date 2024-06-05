from jax import config

config.update("jax_enable_x64", True)

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit

v_low,v_high = -50.0,50.0
av_low,av_high = -2*jnp.pi,2*jnp.pi

va_low,va_high = -35.0,35.0
ava_low,ava_high = -1*jnp.pi,1*jnp.pi

UNI_SI_U_LIM = jnp.array([[v_low,av_low],[v_high,av_high]])
UNI_DI_U_LIM = jnp.array([[va_low,ava_low],[va_high,ava_high]])

# FIX THIS LATER
# @jit
def unicycle_kinematics_single_integrator(U,unicycle_state,dt):
    # sensor dynamics for unicycle model
    horizon = U.shape[-2]
    p,chi = unicycle_state[...,:3],unicycle_state[...,[3]]

    chi = jnp.expand_dims(chi,axis=-1)
    p =  jnp.expand_dims(p,axis=-2)

    vs,avs = jnp.clip(U[...,0],v_low,v_high),jnp.clip(U[...,[1]],av_low,av_high)

    vs = jnp.expand_dims(vs,axis=-1)
    chi_next = chi + jnp.cumsum(dt * avs,axis=-2)
    chis = jnp.concatenate((chi,chi_next),axis=-2)

    chis_temp = chis[...,:-1,[0]]
    ps_next = p + jnp.cumsum(jnp.concatenate((jnp.cos(chis_temp),
                                               jnp.sin(chis_temp),
                                               jnp.zeros(chis_temp.shape)),axis=-1) * vs * dt,axis=-2)
    ps = jnp.concatenate((p,ps_next),axis=-2)

    return jnp.concatenate((ps,chis),axis=-1)

#
# def accelerate(vel_param, change):
#     v = vel_param[0]
#     v_low = vel_param[1]
#     v_high = vel_param[2]
#
#     v_new = jnp.clip(v + change, v_low, v_high)
#
#     return jnp.array([v_new[0], v_low, v_high]), v_new

# @jit
# def unicycle_kinematics_double_integrator(U, unicycle_state, dt):
#     horizon = U.shape[0]
#
#     # sensor dynamics for unicycle model
#     p, chi,v,av = unicycle_state[:, :3], unicycle_state[:, 3], unicycle_state[:,4],unicycle_state[:,5]
#
#
#     a = jnp.clip(U[:,[0]],va_low,va_high)
#     aa = jnp.clip(U[:,[1]],ava_low,ava_high)
#
#     _,vs = accelerate(jnp.array([v[0],v_low,v_high]),jnp.cumsum(a*dt))
#     vs = vs.reshape(-1,1)
#
#     _,avs = accelerate(jnp.array([av[0],av_low,av_high]),jnp.cumsum(aa*dt))
#     avs = avs.reshape(-1,1)
#
#     # vs = jnp.clip(jnp.clip(v,v_low,v_high) + jnp.cumsum(a*dt) ,v_low,v_high).reshape(-1,1)
#     # avs = jnp.clip(jnp.clip(av,av_low,av_high) + jnp.cumsum(aa*dt),av_low,av_high).reshape(-1,1)
#
#     chi = chi.reshape(1, 1)
#     p = p.reshape(1, -1)
#     v = v.reshape(1,-1)
#     av = av.reshape(1,-1)
#
#     chi_next = chi + jnp.cumsum(dt * avs, axis=0)
#     chis = jnp.vstack((chi, chi_next))
#
#     ps_next = p + jnp.cumsum(jnp.column_stack((jnp.cos(chis[:-1].ravel()),
#                                                jnp.sin(chis[:-1].ravel()),
#                                                jnp.zeros(horizon))) * vs * dt, axis=0)
#     ps = jnp.vstack((p, ps_next))
#
#     vs = jnp.vstack((v,vs))
#     avs = jnp.vstack((av,avs))
#
#     return jnp.column_stack((ps, chis,vs,avs))
#
# def accelerate(vel_param, change):
#     v = vel_param[0]
#     v_low = vel_param[1]
#     v_high = vel_param[2]
#
#     v_new = jnp.clip(v + change, v_low, v_high)
#
#     return (v_new, v_low, v_high), v_new
#
# @jit
# def unicycle_kinematics_double_integrator(U, unicycle_state, dt):
#     horizon = U.shape[-2]
#
#     # sensor dynamics for unicycle model
#     p, chi,v,av = unicycle_state[..., :3], unicycle_state[..., [3]], unicycle_state[...,[4]],unicycle_state[...,[5]]
#
#
#     a = jnp.clip(U[...,[0]],va_low,va_high)
#     aa = jnp.clip(U[...,[1]],ava_low,ava_high)
#
#     a = jnp.moveaxis(a,source=-2,destination=0)
#     aa = jnp.moveaxis(aa,source=-2,destination=0)
#
#     # _,vs = accelerate((v,v_low,v_high),jnp.cumsum(a*dt,axis=-2)[...,-1])
#     _,vs = jax.lax.scan(accelerate, (v, v_low, v_high), (a * dt))
#
#     vs = jnp.swapaxes(vs,0,-2)[...,-1]
#
#     # _,avs = accelerate((av,av_low,av_high),jnp.cumsum(aa*dt,axis=-2)[...,-1])
#     _,avs = jax.lax.scan(accelerate, (av,av_low,av_high), (aa * dt))
#
#     avs = jnp.swapaxes(avs,0,-2)[...,-1]
#
#     vs =  jnp.concatenate((v,vs),axis=-1)
#     avs = jnp.concatenate((av,avs),axis=-1)
#
#     chi_next = chi + jnp.cumsum(dt * avs[...,:-1], axis=-1) + jnp.moveaxis(aa,source=0,destination=-1) * dt**2
#
#     chis = jnp.concatenate((chi, chi_next),-1)
#
#     # get chi 1 to k
#     chis_temp = chis[..., :-1]
#     vs_temp = vs[...,:-1]
#     p_change = jnp.cumsum(jnp.stack((jnp.cos(chis_temp)  * vs_temp * dt + jnp.moveaxis(a,source=0,destination=-1) * jnp.cos(chis_temp) * dt**2,
#                jnp.sin(chis_temp) * vs_temp * dt +  jnp.moveaxis(a,source=0,destination=-1) * jnp.sin(chis_temp) * dt**2,
#                jnp.zeros(chis_temp.shape)), axis=-1),axis=-2)
#
#     ps_next = jnp.expand_dims(p,axis=-2) + p_change
#
#     ps = jnp.concatenate((jnp.expand_dims(p,axis=-2), ps_next),axis=-2)
#
#
#
#     chis = jnp.expand_dims(chis,axis=-1)
#     vs = jnp.expand_dims(vs,axis=-1)
#     avs = jnp.expand_dims(avs,axis=-1)
#
#     return jnp.concatenate((ps, chis,vs,avs),axis=-1)




def next_state_fn(current_state, control_input):

    current_pos_x = current_state[...,[0]]
    current_pos_y = current_state[...,[1]]
    current_pos_z = current_state[...,[2]]

    current_ang = current_state[...,[3]]
    current_vel = current_state[...,[4]]
    current_ang_vel = current_state[...,[5]]
    dt = current_state[...,[6]]

    input_acc = jnp.clip(control_input[...,[0]],va_low,va_high)
    input_ang_acc = jnp.clip(control_input[...,[1]],ava_low,ava_high)

    next_vel = current_vel + dt*input_acc
    next_ang_vel = current_ang_vel + dt*input_ang_acc

    next_pos_x = current_pos_x + jnp.cos(current_ang)*current_vel*dt + jnp.cos(current_ang)*.5*input_acc*dt**2
    next_pos_y = current_pos_y + jnp.sin(current_ang)*current_vel*dt + jnp.sin(current_ang)*.5*input_acc*dt**2
    next_pos_z = current_pos_z

    next_ang = current_ang + current_ang_vel*dt + .5 * input_ang_acc*dt**2

    next_vel = jnp.clip(next_vel, v_low, v_high)
    next_ang_vel = jnp.clip(next_ang_vel, av_low, av_high)

    next_state = jnp.concatenate((next_pos_x, next_pos_y, next_pos_z, next_ang, next_vel, next_ang_vel, dt), axis=-1)

    return next_state, next_state

@jit
def unicycle_kinematics_double_integrator(U, unicycle_state, dt):
    horizon = U.shape[-2]

    U = jnp.moveaxis(U,source=-2,destination=0)

    dt_shape = unicycle_state.shape[:-1] + (1,)
    dt_tile = jnp.tile(dt,dt_shape)
    current_state = jnp.concatenate((unicycle_state,dt_tile),axis=-1)

    last_state,next_states = jax.lax.scan(next_state_fn, current_state, U)

    next_states = jnp.moveaxis(next_states[...,:-1],source=0,destination=-2)

    all_states = jnp.concatenate((jnp.expand_dims(unicycle_state,-2),next_states), axis=-2)

    return all_states