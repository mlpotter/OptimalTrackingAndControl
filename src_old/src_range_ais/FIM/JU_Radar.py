
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
import numpy as np
from scipy.spatial import distance_matrix
from jax.tree_util import Partial as partial

from copy import deepcopy

from src_range.control.Sensor_Dynamics import state_multiple_update


@jit
def JU_FIM_radareqn_target_logdet(ps,qs,J,
                               A,Q,
                               Pt,Gt,Gr,L,lam,rcs,c,B,alpha):

    # FIM of single target, multiple sensors

    FIM = JU_FIM_D_Radar(ps=ps,q=qs,J=J,
                         A=A,Q=Q,
                         Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha)

    # sign,logdet = jnp.linalg.slogdet(jnp.linalg.inv(FIM)+jnp.eye(FIM.shape[0])*1e-5)
    # logdet = -logdet
    sign,logdet = jnp.linalg.slogdet(FIM)
    return logdet
#
@jit
def JU_FIM_D_Radar(ps,q,J,A,Q,Pt,Gt,Gr,L,lam,rcs,c,B,alpha):
    q = q.reshape(1,-1)
    N,dn= ps.shape
    _,dm = q.shape

    ps = jnp.concatenate((ps,jnp.zeros((N,1))),-1)
    q = q[:,:dm//2]

    # Qinv = jnp.linalg.inv(Q+jnp.eye(dm)*1e-8)
    # # Qinv = jnp.linalg.inv(Q)
    #
    # D11 = A.T @ Qinv @ A
    # D12 = -A.T @ Qinv
    # D21 = D12.T

    d = (q[jnp.newaxis,:,:] - ps[:,jnp.newaxis,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L
    C = c**2 / (alpha*B**2) * 1/K

    coef = jnp.sqrt((1/(C*distances**6) + 8/(distances**4)))
    outer_vector = d * coef
    outer_product = (outer_vector.transpose(1,2,0) @ outer_vector.transpose(1,0,2))

    # append zeros because there is no velocity in the radar equation...

    # zeros = jnp.zeros_like(d)
    # d = jnp.concatenate((d,zeros),-1)

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    # jnp.einsum("ijk,ilm->ikm", d, d)

    # D22 = jnp.sum(outer_product,axis=0) + Qinv

    # J = D22 - D21 @ jnp.linalg.inv(J + D11) @ D12
    J = jnp.sum(outer_product, axis=0)

    return J


@jit
def FIM_radareqn_target_logdet(ps,qs,
                               Pt,Gt,Gr,L,lam,rcs,c,B,alpha):

    # FIM of single target, multiple sensors

    FIM = FIM_D_Radar(ps=ps,qs=qs,
                         Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha)

    # sign,logdet = jnp.linalg.slogdet(jnp.linalg.inv(FIM)+jnp.eye(FIM.shape[0])*1e-5)
    # logdet = -logdet
    sign,logdet = jnp.linalg.slogdet(FIM)
    return logdet

@jit
def FIM_D_Radar(ps,qs,Pt,Gt,Gr,L,lam,rcs,c,B,alpha):
    N,dn= ps.shape
    M,dm = qs.shape

    d = (qs[jnp.newaxis,:,:] - ps[:,jnp.newaxis,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L

    C = c**2 / (alpha*B**2) * 1/K

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    # jnp.einsum("ijk,ilm->ikm", d, d)
    coef = jnp.sqrt((1/(C*distances**6) + 8/(distances**4)))
    outer_vector = d * coef
    outer_product = (outer_vector.transpose(1,2,0) @ outer_vector.transpose(1,0,2))


    J = jax.scipy.linalg.block_diag(*[outer_product[m] for m in range(M)])
    return J

def Multi_FIM_Logdet_decorator_MPC(score_fn,method="action"):

    # the lower this value, the better!

    if method=="action":
        @jit
        def Multi_FIM_Logdet(U,chis,ps,qs,time_step_sizes,Js,paretos,
                             A,Q,
                             Pt,Gt,Gr,L,lam,rcs,c,B,alpha,
                             gamma):
            horizon = U.shape[1]
            M,dm = qs.shape
            N,dn = ps.shape
            # ps = jnp.expand_dims(ps,1)

            ps,chis,ps_trajectory,chis_trajectory = vmap(state_multiple_update,(0,0,0,0))(ps,U,chis,time_step_sizes)

            multi_FIM_obj = 0

            Js = jnp.stack(Js)

            fim_logdet_parallel = vmap(partial(score_fn,A=A,Q=Q,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha),in_axes=(None,0,0))
            Js_update_parallel = vmap(partial(JU_FIM_D_Radar,A=A,Q=Q,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha),in_axes=(None,0,0))
            # iterate through time step
            for t in jnp.arange(1,horizon+1):
                # iterate through each FIM corresponding to a target
                # for m in range(M):
                #     Jm = Js[m]
                multi_FIM_obj += jnp.sum(gamma**(t-1) * paretos * fim_logdet_parallel(ps_trajectory[:,t],qs,Js))

                                             # paretos[m] * FIM_logdet(ps=ps_trajectory[:,t].squeeze(),qs=qs[[m],:],J=Jm,
                                             #                                A=A,Q=Q,
                                             #                                Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,s=s)
                    #
                    # Js[m] = JU_FIM_D_Radar(ps=ps_trajectory[:,t].squeeze(), q=qs[[m],:], J=Jm,
                    #                        A=A, Q=Q,
                    #                        Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,s=s)
                Js = Js_update_parallel(ps_trajectory[:,t],qs,Js)

                qs = (A @ qs.reshape(-1, dm).T).T.reshape(M, dm)

            return -multi_FIM_obj

    elif method=="FIM2D":
        @jit
        def Multi_FIM_Logdet(ps, qs,
                             Pt, Gt, Gr, L, lam, rcs,c,B,alpha):

            M, dm = qs.shape
            N, dn = ps.shape
            # ps = jnp.expand_dims(ps,1)

            multi_FIM_obj = score_fn(ps,qs, Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,c=c,B=B,alpha=alpha)

            return -multi_FIM_obj


    return Multi_FIM_Logdet

@partial(jit,static_argnames=['N',"space"])
def FIM_2D_Visualization(ps,qs,
                      Pt,Gt,Gr,L,lam,rcs,c,B,alpha,N,space):


    sensor_and_targets = jnp.vstack((ps,qs))

    x_max,y_max = jnp.min(sensor_and_targets,axis=0)-space
    x_min,y_min = jnp.max(sensor_and_targets,axis=0)+space

    qx,qy = jnp.meshgrid(jnp.linspace(x_min,x_max,N),
                         jnp.linspace(y_min,y_max,N))

    qs = jnp.column_stack((qx.ravel(),qy.ravel()))

    d = (qs[:,jnp.newaxis,:] - ps[jnp.newaxis,:,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L

    C = c**2 / (alpha*B**2) * 1/K

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    # jnp.einsum("ijk,ilm->ikm", d, d)
    coef = jnp.sqrt((1/(C*distances**6) + 8/(distances**4)))

    outer_vector = d * coef

    J = (outer_vector.transpose(0,2,1) @ outer_vector)




    # sign,logdet = jnp.linalg.slogdet(jnp.linalg.inv(J)+jnp.tile(jnp.eye(J.shape[-1]),(N**2,1,1))*1e-8)

    # logdet =  -logdet
    sign,logdet = jnp.linalg.slogdet(J)

    return qx,qy,logdet.reshape(N,N)



@partial(jit,static_argnames=['N'])
def FIM_Visualization(ps,qs,
                      Pt,Gt,Gr,L,lam,rcs,s,N):


    sensor_and_targets = jnp.vstack((ps,qs[:,:2]))

    x_max,y_max = jnp.min(sensor_and_targets,axis=0)-30
    x_min,y_min = jnp.max(sensor_and_targets,axis=0)+30

    qx,qy = jnp.meshgrid(jnp.linspace(x_min,x_max,N),
                         jnp.linspace(y_min,y_max,N))

    qs = jnp.column_stack((qx.ravel(),qy.ravel()))

    d = (qs[:,jnp.newaxis,:] - ps[jnp.newaxis,:,:])

    distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))

    K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L

    constant = jnp.sqrt(4 / (distances**8 * s**2 ) * K**2 / (K + s**2 * distances**4 ))

    outer_vector = jnp.expand_dims(d * constant,-1)

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    outer_product =  outer_vector @ outer_vector.transpose(0,1,3,2)

    J = jnp.sum(outer_product, axis=1)

    sign,logdet = jnp.linalg.slogdet(jnp.linalg.inv(J)+jnp.tile(jnp.eye(J.shape[-1]),(N**2,1,1))*1e-8)

    logdet =  -logdet

    return qx,qy,logdet.reshape(N,N)
