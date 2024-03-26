from jax import config

config.update("jax_enable_x64", True)

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit

v_low,v_high = -50,50
av_low,av_high = -2*jnp.pi,2*jnp.pi

va_low,va_high = -35,35
ava_low,ava_high = -1*jnp.pi,1*jnp.pi

UNI_SI_U_LIM = jnp.array([[v_low,av_low],[v_high,av_high]])
UNI_DI_U_LIM = jnp.array([[va_low,ava_low],[va_high,ava_high]])

@jit
def unicycle_kinematics_single_integrator(U,unicycle_state,time_step_size):
    # sensor dynamics for unicycle model
    p,chi = unicycle_state[:,:2],unicycle_state[:,2]

    vs,avs = jnp.clip(U[:,[0]],v_low,v_high),jnp.clip(U[:,[1]],av_low,av_high)

    chi = chi.reshape(1,1)
    p = p.reshape(1,-1)

    chi_next = chi + jnp.cumsum(time_step_size * avs,axis=0)
    chis = jnp.vstack((chi,chi_next))

    ps_next = p + jnp.cumsum(jnp.column_stack((jnp.cos(chis[:-1].ravel()),
                                               jnp.sin(chis[:-1].ravel()))) * vs * time_step_size,axis=0)
    ps = jnp.vstack((p,ps_next))

    return jnp.column_stack((ps,chis))


def accelerate(vel_param, change):
    v = vel_param[0]
    v_low = vel_param[1]
    v_high = vel_param[2]

    v_new = jnp.clip(v + change, v_low, v_high)

    return jnp.array([v_new[0], v_low, v_high]), v_new

@jit
def unicycle_kinematics_double_integrator(U, unicycle_state, time_step_size):
    # sensor dynamics for unicycle model
    p, chi,v,av = unicycle_state[:, :2], unicycle_state[:, 2], unicycle_state[:,3],unicycle_state[:,4]


    a = jnp.clip(U[:,[0]],va_low,va_high)
    aa = jnp.clip(U[:,[1]],ava_low,ava_high)

    _,vs = accelerate(jnp.array([v[0],v_low,v_high]),jnp.cumsum(a*time_step_size))
    vs = vs.reshape(-1,1)

    _,avs = accelerate(jnp.array([av[0],av_low,av_high]),jnp.cumsum(aa*time_step_size))
    avs = avs.reshape(-1,1)

    # vs = jnp.clip(jnp.clip(v,v_low,v_high) + jnp.cumsum(a*time_step_size) ,v_low,v_high).reshape(-1,1)
    # avs = jnp.clip(jnp.clip(av,av_low,av_high) + jnp.cumsum(aa*time_step_size),av_low,av_high).reshape(-1,1)

    chi = chi.reshape(1, 1)
    p = p.reshape(1, -1)
    v = v.reshape(1,-1)
    av = av.reshape(1,-1)

    chi_next = chi + jnp.cumsum(time_step_size * avs, axis=0)
    chis = jnp.vstack((chi, chi_next))

    ps_next = p + jnp.cumsum(jnp.column_stack((jnp.cos(chis[:-1].ravel()),
                                               jnp.sin(chis[:-1].ravel()))) * vs * time_step_size, axis=0)
    ps = jnp.vstack((p, ps_next))

    vs = jnp.vstack((v,vs))
    avs = jnp.vstack((av,avs))

    return jnp.column_stack((ps, chis,vs,avs))