from jax import config

config.update("jax_enable_x64", True)

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit


# def rotational_matrix(chi):
#     return jnp.array([[jnp.cos(chi), -jnp.sin(chi)],
#                       [jnp.sin(chi), jnp.cos(chi) ]])
# def rotational_column_perp(chi):
#     return jnp.vstack((-jnp.sin(chi),jnp.cos(chi)))
#
# def rotational_column(chi):
#     return jnp.vstack((jnp.cos(chi),jnp.sin(chi)))
#
# # @jit
# def angle_update(chi,av,time_step_size):
#     return chi + time_step_size*av
#
# def position_update(p,v,av,chi,chi_,time_step_size):
#
#     p = p.T
#
#     return jnp.where(av==0,
#                      p + v *rotational_column(chi)*time_step_size,
#                      p + (v / jnp.where(av == 0., 1e-10, av)) * (rotational_column_perp(chi) - rotational_column_perp(chi_))
#                      ).T
#
# @jit
# def state_multiple_update(p,U,chi,time_step_sizes):
#     vs,avs = U[0,:],U[1,:]
#     chis = [jnp.expand_dims(chi,0)] + [None]*len(vs)
#     ps = [jnp.expand_dims(p,0)] + [None]*len(vs)
#
#
#
#     for k in range(len(vs)):
#         # update angle
#         chi_next = angle_update(chi,avs[k],time_step_sizes[k])
#         chis[k+1] = jnp.expand_dims(chi_next, 0)
#
#         # update position on angle
#         p_next = position_update(p,vs[k],avs[k],chi,chi_next,time_step_sizes[k])
#         ps[k+1] = jnp.expand_dims(p_next,0)
#
#         # reinit for next state
#         chi = chi_next
#         p = p_next
#
#     return p,chi,jnp.vstack(ps),jnp.vstack(chis)

# @jit
def state_multiple_update(p,U,chi,time_step_sizes):
    # sensor dynamics for unicycle model

    vs,avs = U[:,[0]],U[:,[1]]

    chi = chi.reshape(1,1)
    p = p.reshape(1,-1)

    chi_next = chi + jnp.cumsum(time_step_sizes * avs,axis=0)
    chis = jnp.vstack((chi,chi_next))

    ps_next = p + jnp.cumsum(jnp.column_stack((jnp.cos(chis[:-1].ravel()),
                                               jnp.sin(chis[:-1].ravel()))) * vs * time_step_sizes,axis=0)
    ps = jnp.vstack((p,ps_next))

    # chis = [jnp.expand_dims(chi,0)] + [None]*len(vs)
    # ps = [jnp.expand_dims(p,0)] + [None]*len(vs)
    #
    # for k in range(len(vs)):
    #     chi_next = chi + time_step_sizes * avs[k]
    #     p_next = p + time_step_sizes * jnp.array([[jnp.cos(chi.squeeze()),jnp.sin(chi.squeeze())]]) * vs[k]
    #
    #     ps[k+1] = jnp.expand_dims(p_next,0)
    #     chis[k+1] = jnp.expand_dims(chi_next,0)
    #
    #     chi = chi_next
    #     p = p_next
    # chis = jnp.hstack((chi,chi+jnp.cumsum(time_)))
    # p = p.reshape(-1,2)
    # chi = chi.reshape(1,1)

    return ps[-1,:],chis[-1,:],ps,chis
