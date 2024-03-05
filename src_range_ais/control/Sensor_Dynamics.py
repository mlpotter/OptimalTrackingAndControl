from jax import config

config.update("jax_enable_x64", True)

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit

def unicycle_kinematics(p,U,chi,time_step_sizes):
    # sensor dynamics for unicycle model
    v_low,v_high = 0,50
    av_low,av_high = -jnp.pi,jnp.pi
    vs,avs = jnp.clip(U[:,[0]],v_low,v_high),jnp.clip(U[:,[1]],av_low,av_high)

    chi = chi.reshape(1,1)
    p = p.reshape(1,-1)

    chi_next = chi + jnp.cumsum(time_step_sizes * avs,axis=0)
    chis = jnp.vstack((chi,chi_next))

    ps_next = p + jnp.cumsum(jnp.column_stack((jnp.cos(chis[:-1].ravel()),
                                               jnp.sin(chis[:-1].ravel()))) * vs * time_step_sizes,axis=0)
    ps = jnp.vstack((p,ps_next))

    return ps[-1,:],chis[-1,:],ps,chis
