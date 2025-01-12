import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import imageio
import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from tqdm import tqdm
from time import time
from copy import deepcopy

import os
import glob

from src_power.FIM.JU_Radar import *
from src_power.utils import NoiseParams

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds


config.update("jax_enable_x64", True)

if __name__ == "__main__":



    seed = 555
    key = jax.random.PRNGKey(seed)

    # Experiment Choice
    update_steps = 0
    FIM_choice = "radareqn"
    measurement_choice = "radareqn"
    method = 'FIM2D'

    # Save frames as a GIF
    pdf_filename = "radar_optimal_RICE.pdf"
    pdf_savepath = os.path.join("..", "..", "images")
    photo_dump = os.path.join("tmp_images")
    remove_photo_dump = True
    os.makedirs(photo_dump, exist_ok=True)


    Restarts = 25
    N = 5

    # ==================== RADAR CONFIGURATION ======================== #
    speedoflight = 299792458
    fc = 1e9;
    Gt = 2000;
    Gr = 2000;
    lam = speedoflight / fc
    rcs = 1;
    L = 1;

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 100

    K = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    coef = K / (R ** 4)


    SCNR = -20
    CNR = -10
    Pt = 10000
    Amp, Ma, zeta, s = NoiseParams(Pt * coef, SCNR, CNR=CNR)

    print("Spread: ",s**2)
    print("Power Return (RCS): ",coef*Pt)
    print("K",K)

    print("Pt (peak power)={:.9f}".format(Pt))
    print("lam ={:.9f}".format(lam))

    key_args = {"Pt": Pt, "Gt": Gt, "Gr": Gr, "lam": lam, "L": L, "rcs": rcs, "R": 100,"SCNR":SCNR,"CNR":CNR,"s":s}

    # ==================== SENSOR CONSTRAINTS ======================== #
    R_sensors_to_targets = 500.
    R_sensors_to_sensors = 250


    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-2000, maxval=2000)
    qs = jnp.array([
                    [-250,-250.], #,
                    [250,250]]) #,
                    # [50,-20]])

    M, dm = qs.shape;
    N ,dn = ps.shape;

    FIM_D_Radar(ps, qs=qs,
                   Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,s=s)

    Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC(FIM_radareqn_target_logdet,method=method)


    constraints = []
    def distance_constraint_sensors_to_targets(ps_optim):
        ps_optim = ps_optim.reshape(N,dn)
        difference = (qs[jnp.newaxis, :,:] - ps_optim[:, jnp.newaxis, :])
        distance = jnp.sqrt(jnp.sum(difference ** 2, -1))

        return distance.ravel()

    def distance_constraint_sensors_to_sensors(ps_optim):
        ps_optim = ps_optim.reshape(N,dn)
        idx = np.arange(N)[:,None] < np.arange(N)
        difference = (ps_optim[jnp.newaxis, :,:] - ps_optim[:, jnp.newaxis, :])
        distance = jnp.sqrt(jnp.sum(difference ** 2, -1))

        return distance[idx].ravel()


    constraints.append(NonlinearConstraint(distance_constraint_sensors_to_targets,R_sensors_to_targets/2,np.inf))
    constraints.append(NonlinearConstraint(distance_constraint_sensors_to_sensors,R_sensors_to_sensors/2,np.inf))

    Multi_FIM_Logdet_partial = partial(Multi_FIM_Logdet,qs=qs,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,s=s)
    objective = lambda ps_optim: Multi_FIM_Logdet_partial(ps_optim.reshape(N,dn))

    jac_jax = jax.jit(jax.grad(objective,argnums=0))
    hess_jax = jax.jit(jax.hessian(objective,argnums=0))

    jac = lambda ps_optim: jac_jax(ps_optim).ravel()
    hess = lambda ps_optim: hess_jax(ps_optim)

    f_best = jnp.inf
    ps_best = 0
    for k in range(Restarts):
        print("="*10,f"initialization {k}", "="*10)
        print("Obj Init",objective(ps))


        SLSQP = minimize(fun=objective, x0=ps.ravel(),method="trust-constr",constraints=constraints,options={"maxiter":10000,"disp":True},jac=jac,hess=hess)
        print(SLSQP)
        ps_optim = SLSQP.x.reshape(N,dn)
        # ps = ps_optim

        J = FIM_D_Radar(ps_optim, qs, Pt, Gt, Gr, L, lam, rcs, s)
        print("Matrix Rank: ",jnp.linalg.matrix_rank(J))
        if f_best > SLSQP.fun:
            f_best = SLSQP.fun
            ps_best = ps_optim

        print("\n")

        key, subkey = jax.random.split(key)

        ps = jax.random.uniform(key, shape=(N, 2), minval=-2000, maxval=2000)
        print("BEST OBJECTIVE IS: ",f_best)

    # ps_best = jnp.array([[-450,-480],
    #                      [-450,480],
    #                      [0,250],
    #                      [-250,0],
    #                      [450,480],
    #                      [0,-0]])
    # ps_best = jnp.array([[-450,-480],
    #                      [450,480],
    #                      [0,250],
    #                      [-250,0],
    #                      [0,-0]])
    best_obj = objective(ps_best)
    print("Best Objective is: ",best_obj)





    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #
    # for m in range(M):
    #     axes.add_patch(Circle(qs[m, :2], R_sensors_to_targets, edgecolor="red", fill=False, lw=1, linestyle="--"))
    axes[0].plot(qs[:,0],qs[:,1],"mo",label="Target")
    for m in range(M):
        axes[0].add_patch(
            Circle(qs[m,:], R_sensors_to_targets / 2, edgecolor="magenta", fill=False, lw=1,
                   linestyle="--",label="_nolegend_"))

    for n in range(N):
        axes[0].add_patch(
            Circle(ps_best[n,:], R_sensors_to_sensors / 2, edgecolor="red", fill=False, lw=1,
                   linestyle="--",label="_nolegend_"))

    axes[0].axis('equal')
    axes[0].plot([],[],"r--",label="Radar Boundary")
    axes[0].plot([],[],"m--",label="Target Boundary")
    axes[0].plot(ps_best[:,0],ps_best[:,1],"ro",label="Radar")
    axes[0].plot(ps[:,0],ps[:,1],"rX")
    axes[0].set_title(f"Best Log |J| = -{np.round(best_obj,5)}")
    axes[0].legend()

    qx, qy, logdet_grid = FIM_2D_Visualization(ps=ps_best, qs=qs,
                                            Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, s=s,
                                            N=2500)

    CS = axes[1].contourf(qx, qy, logdet_grid, levels=15)
    # pcm = axes[1].pcolor(qx,qy,logdet_grid)
    axes[1].scatter(ps_best[:, 0], ps_best[:, 1], s=50, marker="o", color="r")
    #
    axes[1].scatter(qs[:, 0], qs[:, 1], s=50, marker="o", color="m")
    axes[1].set_title("log |J|")

    fig.colorbar(CS,ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(pdf_savepath,pdf_filename))
    fig.show()
    plt.minorticks_off()

#     sensor_and_targets = jnp.vstack((ps,qs[:,:2]))
#
#     x_max,y_max = jnp.min(sensor_and_targets,axis=0)-30
#     x_min,y_min = jnp.max(sensor_and_targets,axis=0)+30
#
#     qx,qy = jnp.meshgrid(jnp.linspace(x_min,x_max,N),
#                          jnp.linspace(y_min,y_max,N))
#
#     qs = jnp.column_stack((qx.ravel(),qy.ravel()))
#
#     d = (qs[:,jnp.newaxis,:] - ps[jnp.newaxis,:,:])
#
#     distances = jnp.sqrt(jnp.sum(d**2,-1,keepdims=True))
#
#     K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L
#
#     constant = jnp.sqrt(4 / (distances**8 * s**2 ) * K**2 / (K + s**2 * distances**4 ))
#
#     outer_vector = jnp.expand_dims(d * constant,-1)
#
#     # dd^T / ||d||^4 * rho * rho/(rho+1)
#     outer_product =  outer_vector @ outer_vector.transpose(0,1,3,2)
#
#     J = jnp.sum(outer_product, axis=1)
#
#     sign,logdet = jnp.linalg.slogdet(jnp.linalg.inv(J)+jnp.tile(jnp.eye(J.shape[-1]),(N**2,1,1))*1e-8)
#
#     logdet =  -logdet
#
#     return qx,qy,logdet.reshape(N,N)
