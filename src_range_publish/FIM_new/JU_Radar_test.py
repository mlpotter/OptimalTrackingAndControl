import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxopt import ScipyMinimize,ScipyBoundedMinimize
import numpy as np

import imageio
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from tqdm import tqdm
from time import time
from copy import deepcopy

import os
import glob

from src_range_publish.FIM_new.FIM_RADAR import *
from src_range_publish.objective_fns.objectives import *


config.update("jax_enable_x64", True)

if __name__ == "__main__":



    seed = 555
    key = jax.random.PRNGKey(seed)
    np.random.seed(555)

    # Experiment Choice
    update_steps = 0
    FIM_choice = "radareqn"
    measurement_choice = "radareqn"
    mpc_method = "Single_FIM_3D_action"
    fim_method = "Standard FIM"

    # Save frames as a GIF
    gif_filename = "radar_optimal_RICE.gif"
    gif_savepath = os.path.join("..", "..", "images","gifs")
    photo_dump = os.path.join("tmp_images")
    remove_photo_dump = True
    os.makedirs(photo_dump, exist_ok=True)

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = .1
    NT = 250
    N = 6

    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e9;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    alpha = (jnp.pi)**2 / 3
    B = 0.05 * 10**5


    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000

    Pt = 10000
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    Pr = K / (R ** 4)

    # get the power of the noise of the signal
    SNR=0



    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    time_steps = 15
    dt = 0.1
    control_constraints = UNI_DI_U_LIM
    kinematic_model = unicycle_kinematics_double_integrator


    # ==================== MPPI CONFIGURATION ================================= #
    # ps = place_sensors([-100,100],[-100,100],N)
    key, subkey = jax.random.split(key)
    #
    ps = jnp.concatenate((jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100),jnp.zeros((N,1))),axis=-1)
    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    vs = jnp.zeros((ps.shape[0],1))
    avs = jnp.zeros((ps.shape[0],1))
    radar_state = jnp.column_stack((ps,chis,vs,avs))

    ps_init = deepcopy(ps)
    # qs = jnp.array([[0.0, -0.0, 25., 20], #,#,
    #                 [-50.4,30.32,-20,-10], #,
    #                 [10,10,10,10],
    #                 [20,20,5,-5]])
    z_elevation=150
    qs = jnp.array([[0.0, -0.0,z_elevation, 15., 10,0], #,#,
                    [-50.4,30.32,z_elevation,-10,-5,0], #,
                    [10,10,z_elevation,5,5,0],
                    [20,20,z_elevation,2.5,-2.5,0]])

    M, dm = qs.shape;
    N, dn = ps.shape;

    # ======================== MPC Assumptions ====================================== #
    gamma = 0.95
    R_sensors_to_targets = 25
    R_sensors_to_sensors = 10


    sigmaQ = jnp.sqrt(10 ** 0);
    sigmaW = jnp.sqrt(M*Pr/ (10**(SNR/10)))

    C = c**2 * sigmaW**2 / (jnp.pi**2 * 8 * fc**2) * 1/K

    print("SigmaQ (state noise)={}".format(sigmaQ))

    print("Power Return (RCS): ",Pr)
    print("Noise Power: ",sigmaW**2)
    print("K",K)

    print("Pt (peak power)={:.9f}".format(Pt))
    print("lam ={:.9f}".format(lam))
    print("C=",C)

    # A_single = jnp.array([[1., 0, T, 0],
    #                       [0, 1., 0, T],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1.]])
    #
    # Q_single = jnp.array([
    #     [(T ** 4) / 4, 0, (T ** 3) / 2, 0],
    #     [0, (T ** 4) / 4, 0, (T ** 3) / 2],
    #     [(T ** 3) / 2, 0, (T ** 2), 0],
    #     [0, (T ** 3) / 2, 0, (T ** 2)]
    # ]) * sigmaQ ** 2

    A_single = jnp.array([[1., 0, 0, dt, 0, 0],
                   [0, 1., 0, 0, dt, 0],
                   [0, 0, 1, 0, 0, dt],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(dt ** 4) / 4, 0, 0, (dt ** 3) / 2, 0, 0],
        [0, (dt ** 4) / 4, 0, 0, (dt** 3) / 2, 0],
        [0, 0, (dt**4)/4, 0, 0, (dt**3) / 2],
        [(dt ** 3) / 2, 0, 0, (dt ** 2), 0, 0],
        [0, (dt ** 3) / 2, 0, 0, (dt ** 2), 0],
        [0, 0, (dt**3) / 2, 0, 0, (dt**2)]
    ]) * sigmaQ ** 2

    A = jnp.kron(jnp.eye(M), A_single);
    Q = jnp.kron(jnp.eye(M), Q_single);  # + np.eye(M*Q_single.shape[0])*1e-1;
    G = jnp.eye(N)

    nx = Q.shape[0]

    # Js = jnp.stack([jnp.eye(d) for m in range(M)])
    J = jnp.eye(dm*M) #jnp.stack([jnp.eye(d) for m in range(M)])

    Qinv = jnp.linalg.solve(Q + jnp.eye(nx)**1e-4,jnp.eye(nx))#jnp.linalg.inv(Q+jnp.eye(dm*M)*1e-8)

    if fim_method == "PCRLB":
        IM_fn = partial(Single_JU_FIM_Radar, A=A, Qinv=Qinv, C=C)

    elif fim_method == "Standard FIM":
        IM_fn = partial(Single_FIM_Radar, C=C)


    MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=dt,gamma=gamma,method=mpc_method)

    print("Optimization START: ")
    lbfgsb =  ScipyBoundedMinimize(fun=MPC_obj, method="L-BFGS-B",jit=True)

    # dts = jnp.tile(dt, (N, 1))

    U_upper = (jnp.ones((time_steps, 2)) * control_constraints[1].reshape(1,-1))
    U_lower = (jnp.ones((time_steps, 2)) * control_constraints[0].reshape(1,-1))

    U_lower = jnp.tile(U_lower, jnp.array([N, 1, 1]))
    U_upper = jnp.tile(U_upper, jnp.array([N, 1, 1]))
    bounds = (U_lower, U_upper)

    m0 = qs

    J_list = []
    frames = []
    frame_names = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k in range(NT):
        start = time()
        qs_previous = m0

        m0 = (A @ m0.reshape(-1, 1)).reshape(M, dm)

        U1 = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=control_constraints[0,0]+5, maxval=control_constraints[1,0])
        U2 = jax.random.uniform(key, shape=(N, time_steps, 1 ), minval=control_constraints[1,0],maxval=control_constraints[1,1])
        U = jnp.concatenate((U1, U2), axis=-1)

        # U = jnp.zeros((N,2,time_steps))

        U = lbfgsb.run(U, bounds=bounds, radar_state=radar_state, target_state=m0,
                       J=J,
                       A=A,
                       ).params

        radar_states = kinematic_model( U ,radar_state, dt)

        radar_state = radar_states[:,1]

        print("Vmin: ",radar_states[:, 1][:, 4].min(),"Vmax: ",radar_states[:, 1][:, 4].max())
        # print(ps.shape,chis.shape,ps.squeeze().shape)
        # ps = ps.squeeze()
        # chis = chis.squeeze()


        end = time()
        print(f"Step {k} Optimization Time: ",end-start)

        # m0  = ps

        J = IM_fn(radar_state=radar_state,target_state=m0,J=J) #[JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],s=s) for i in range(len(Js))]

        # print([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        J_list.append(jnp.linalg.slogdet(J)[1].ravel())

        print("FIM (higher is better) ",np.sum(J_list[-1]))

        save_time = time()
        if (k+1)%frame_skip == 0:
            # fig.minorticks_off()

            axes[0].plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="_nolegend_")
            axes[0].plot(m0[:,0], m0[:,1], 'go',label="Targets")
            axes[0].plot(ps_init[:,0], ps_init[:,1], 'md',label="Sensor Init")
            axes[0].plot(radar_state[:,0], radar_state[:,1], 'rx',label="Sensors Next Position")
            axes[0].plot(radar_states[:,0,0], radar_states[:,0,1], 'r*',label="Sensor Position")
            axes[0].plot(radar_states[:,1:,0].T, radar_states[:,1:,1].T, 'r.-',label="_nolegend_")
            axes[0].plot([],[],"r.-",label="Sensor Planned Path")

            axes[0].legend(bbox_to_anchor=(0.7, 0.95))
            axes[0].set_title(f"k={k}")
            axes[0].axis('equal')

            qx,qy,logdet_grid = FIM_Visualization(ps=radar_state[:,:3], qs=m0,
                                                  C=C,
                                                  N=1000)

            axes[1].contourf(qx, qy, logdet_grid, levels=20)
            axes[1].scatter(radar_state[:, 0], radar_state[:, 1], s=50, marker="x", color="r")
            #
            axes[1].scatter(m0[:, 0], m0[:, 1], s=50, marker="o", color="g")
            axes[1].set_title("Instant Time Objective Function Map")
            axes[1].axis('equal')

            axes[2].plot(jnp.array(J_list),"b-",label="Total FIM")
            # axes[2].plot(jnp.array(J_list),"r-",label="Individual FIM")
            axes[2].set_ylabel("Target logdet FIM (Higher is Better)")
            axes[2].set_title(f"Avg MPPI LogDet FIM={np.round(J_list[-1])}")
            axes[2].set_xlabel("Time Step")


            filename = f"frame_{k}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(photo_dump, filename))

            frame_names.append(os.path.join(photo_dump, filename))
            axes[0].cla()
            axes[1].cla()
            axes[2].cla()
        save_time = time() - save_time
        print("Figure Save Time: ",save_time)

    print("lol")


    for frame_name in frame_names:
        frames.append(imageio.imread(frame_name))

    imageio.mimsave(os.path.join(gif_savepath, gif_filename), frames, duration=0.25)
    if remove_photo_dump:
        for filename in glob.glob(os.path.join(photo_dump, "frame_*")):
            os.remove(filename)


