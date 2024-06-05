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

from src_range.FIM_new.FIM_RADAR import *
from src_range.utils import NoiseParams

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
    method = 'Single_FIM_2D_noaction'

    # Save frames as a GIF
    pdf_filename = "radar_optimal_range.pdf"
    pdf_savepath = os.path.join("..", "..", "images")
    photo_dump = os.path.join("tmp_images")
    remove_photo_dump = True
    os.makedirs(photo_dump, exist_ok=True)


    Restarts = 10
    N = 6

    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e6;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    alpha = (jnp.pi)**2 / 3
    B = 0.05 * 10**5
    Pt = 10000
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3



    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000
    Pr = K / (R ** 4)


    SNR = -20
    # calculate Pt such that I achieve SNR=x at distance R=y

    # ==================== SENSOR CONSTRAINTS ======================== #
    R_sensors_to_targets = 750.
    R_sensors_to_sensors = 500


    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-2000, maxval=2000)
    qs = jnp.array([
                    [-2500,-2500.,25,25], #,
                    [2500,2500,-20,6], #,
                    [2000,-2500,-10,-10]])

    M, dm = qs.shape;
    N ,dn = ps.shape;

    sigmaW = jnp.sqrt(M*Pr / (10**(SNR/10)))

    C = c**2 * sigmaW**2 / (jnp.pi**2 * 8 * fc**2) * 1/K

    # print("Power Return (RCS): ",coef*Pt)
    print("K",K)
    print("C",C)
    print("Pt (peak power)={:.9f}".format(Pt))
    print("Noise Power={:.5f}".format(sigmaW**2))
    print("lam ={:.9f}".format(lam))



    IM_fn = partial(Single_FIM_Radar,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW)

    Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC(IM_fn=IM_fn,method=method)

    constraints = []
    def distance_constraint_sensors_to_targets(ps_optim):
        ps_optim = ps_optim.reshape(N,dn)
        difference = (qs[jnp.newaxis, :,:dm//2] - ps_optim[:, jnp.newaxis, :])
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

    Multi_FIM_Logdet_partial = partial(Multi_FIM_Logdet,target_states=qs)
    objective = lambda ps_optim: Multi_FIM_Logdet_partial(radar_states=ps_optim.reshape(N,dn))

    jac_jax = jax.jit(jax.grad(objective,argnums=0))
    hess_jax = jax.jit(jax.hessian(objective,argnums=0))

    jac = lambda ps_optim: jac_jax(ps_optim).ravel()
    hess = lambda ps_optim: hess_jax(ps_optim)

    f_best = jnp.inf
    ps_best = 0
    for k in range(Restarts):
        print("="*10,f"initialization {k}", "="*10)
        print("Obj Init",objective(ps))


        solver = minimize(fun=objective, x0=ps.ravel(),method="SLSQP",constraints=constraints,options={"maxiter":10000,"disp":True},jac=jac,hess=hess)
        print(solver)
        ps_optim = solver.x.reshape(N,dn)
        # ps = ps_optim

        J = IM_fn(radar_states=ps_optim, target_states=qs)
        print("Matrix Rank: ",jnp.linalg.matrix_rank(J))
        if f_best > solver.fun:
            f_best = solver.fun
            ps_best = ps_optim

        print("\n")

        key, subkey = jax.random.split(key)

        ps = jax.random.uniform(key, shape=(N, 2), minval=-2000, maxval=2000)
        print("BEST OBJECTIVE IS: ",f_best)

    # ps_best = jnp.array([[-2149,-2632],
    #                      [2637,2151],
    #                      [3000,3000],
    #                      [2150,2365],
    #                      [-2244,-2300],
    #                      [-2369,-2149]])
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
    axes[0].plot(ps[:,0],ps[:,1],"rX",label="Radar Init")
    axes[0].set_title(f"Best Log |J| = {-np.round(best_obj,5)}")
    axes[0].legend()

    qx, qy, logdet_grid = FIM_2D_Visualization(ps=ps_best, qs=qs[:,:dm//2],
                                            Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs,c=c,fc=fc, sigmaW=sigmaW,
                                            N=2500,space=1000)

    CS = axes[1].contourf(qx, qy, logdet_grid, levels=30)
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