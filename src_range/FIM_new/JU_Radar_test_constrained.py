import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize,NonlinearConstraint,Bounds

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

from src_range.FIM_new.FIM_RADAR import *
from src_range.objective_fns.objectives import *
from src_range.utils import NoiseParams


config.update("jax_enable_x64", True)

if __name__ == "__main__":



    seed = 123
    key = jax.random.PRNGKey(seed)
    np.random.seed(123)

    # Experiment Choice
    update_steps = 0
    FIM_choice = "radareqn"
    measurement_choice = "radareqn"
    method = "Single_FIM_3D_action"
    fim_method = "Standard FIM"

    # Save frames as a GIF
    gif_mpc_filename = "radar_optimal_constrained.gif"
    gif_control_filename = "radar_optimal_constrained_controls.gif"

    gif_savepath = os.path.join("..", "..", "images","gifs")
    photo_dump = os.path.join("tmp_images")
    remove_photo_dump = True
    os.makedirs(photo_dump, exist_ok=True)

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = .1
    NT = 115
    N = 6
    colors = plt.cm.jet(np.linspace(0, 1, N ))

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


    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000

    Pt = 10000
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    Pr = K / (R ** 4)

    # get the power of the noise of the signal
    SNR=0



    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    time_steps = 15
    R_sensors_to_targets = 15.0
    R_sensors_to_sensors = 1.5
    time_step_size = T
    max_velocity = 50.
    min_velocity = 0
    max_angle_velocity = jnp.pi
    min_angle_velocity = -jnp.pi

    # ==================== MPPI CONFIGURATION ================================= #
    limits = jnp.array([[max_velocity, max_angle_velocity], [min_velocity, min_angle_velocity]])

    # ps = place_sensors([-100,100],[-100,100],N)
    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100)
    ps_init = deepcopy(ps)
    # qs = jnp.array([[0.0, -0.0, 25., 20], #,#,
    #                 [-50.4,30.32,-20,-10], #,
    #                 [10,10,10,10],
    #                 [20,20,5,-5]])
    z_elevation=10
    qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
                    [-50.4,30.32,z_elevation,-20,-10,0], #,
                    [10,10,z_elevation,10,10,0],
                    [20,20,z_elevation,5,-5,0]])

    M, dm = qs.shape;
    N, dn = ps.shape;

    theta = jnp.arcsin(z_elevation/R_sensors_to_targets)
    radius_projected = R_sensors_to_targets * jnp.cos(theta)

    # ======================== MPC Assumptions ====================================== #
    gamma = 0.95
    # paretos = jnp.ones((M,)) * 1 / M  # jnp.array([1/3,1/3,1/3])
    # paretos = jnp.array([1/3,1/3,1/3,0,0])

    # assert len(paretos) == M, "Pareto weights not equal to number of targets!"
    # assert (jnp.sum(paretos) <= (1 + 1e-5)) and (jnp.sum(paretos) >= -1e-5), "Pareto weights don't sum to 1!"


    sigmaQ = jnp.sqrt(10 ** 2);
    sigmaV = jnp.sqrt(9)
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

    A_single = jnp.array([[1., 0, 0, T, 0, 0],
                   [0, 1., 0, 0, T, 0],
                   [0, 0, 1, 0, 0, T],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(T ** 4) / 4, 0, 0, (T ** 3) / 2, 0, 0],
        [0, (T ** 4) / 4, 0, 0, (T ** 3) / 2, 0],
        [0, 0, (T**4)/4, 0, 0, (T**3) / 2],
        [(T ** 3) / 2, 0, 0, (T ** 2), 0, 0],
        [0, (T ** 3) / 2, 0, 0, (T ** 2), 0],
        [0, 0, (T**3) / 2, 0, 0, (T**2)]
    ]) * sigmaQ ** 2

    A = jnp.kron(jnp.eye(M), A_single);
    Q = jnp.kron(jnp.eye(M), Q_single);  # + np.eye(M*Q_single.shape[0])*1e-1;
    G = jnp.eye(N)

    nx = Q.shape[0]

    # Js = jnp.stack([jnp.eye(d) for m in range(M)])
    J = jnp.eye(dm*M) #jnp.stack([jnp.eye(d) for m in range(M)])

    Qinv = jnp.linalg.inv(Q+jnp.eye(dm*M)*1e-8)

    if fim_method == "PCRLB":
        IM_fn = partial(Single_JU_FIM_Radar,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW)
    elif fim_method == "Standard FIM":
        IM_fn = partial(Single_FIM_Radar,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW)

    # IM_fn(ps,qs[[0],:],Js=Js)
    IM_fn(ps,qs,J=J)

    # IM_fn_parallel = vmap(IM_fn, in_axes=(None, 0, 0))

    MPC_obj = MPC_decorator(IM_fn=IM_fn,method=method)

    print("Optimization START: ")

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    # time_step_sizes = jnp.tile(time_step_size, (N, 1))

    dc = 2

    U_upper = (jnp.ones((time_steps, dc)) * jnp.array([[max_velocity, max_angle_velocity]]))
    U_lower = (jnp.ones((time_steps, dc)) * jnp.array([[min_velocity, min_angle_velocity]]))

    U_lower = jnp.tile(U_lower, jnp.array([N, 1, 1]))
    U_upper = jnp.tile(U_upper, jnp.array([N, 1, 1]))
    bounds = (U_lower, U_upper)

    U_velocity = jax.random.uniform(key, shape=(N, time_steps, 1), minval=min_velocity + 25, maxval=max_velocity)
    U_angular_velocity = jax.random.uniform(key, shape=(N, time_steps, 1), minval=min_angle_velocity,
                                            maxval=max_angle_velocity)
    U = jnp.concatenate((U_velocity, U_angular_velocity), axis=-1)

    bnds = Bounds(U_lower.ravel(),U_upper.ravel())


    @jit
    def collision_constraint(U,radar_states, target_states):
        _, _, radar_states, chis_trajectory = vmap(unicycle_kinematics, (0, 0, 0, None))(radar_states, U, chis,
                                                                                          time_step_size)
        radar_states = radar_states[:,1:,:]
        N, _, dn = radar_states.shape
        M, T, dm = target_states.shape

        radar_states = jnp.concatenate((radar_states, jnp.zeros((N, T, 1))), -1)
        radar_states = radar_states[:, :, :dm // 2]

        target_positions = target_states[:, :, :dm // 2]

        d = (target_positions[jnp.newaxis, :, :] - radar_states[:, jnp.newaxis, :])

        distances = jnp.sqrt(jnp.sum(d ** 2, -1, keepdims=True))


        return distances.ravel()


    m0 = qs



    J_list = []
    images_mpc = []; images_control = []
    state_multiple_update_parallel = vmap(unicycle_kinematics, (0, 0, 0, None))
    fig_mpc, axes_mpc = plt.subplots(1, 3, figsize=(15, 5))
    fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))

    objective_jac = jit(jax.grad(MPC_obj, argnums=0))

    for k in range(NT):
        start = time()
        qs_previous = m0

        m0 = (A @ m0.reshape(-1, 1)).reshape(M, dm)

        m0_rollout = jnp.stack([(jnp.linalg.matrix_power(A, t - 1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in
                                range(1, time_steps + 1)], axis=1)

        # U = jnp.zeros((N,2,time_steps))
        objective = lambda U: MPC_obj(U=U.reshape(N,time_steps,dc),chis=chis,radar_states=ps,target_states=m0,time_step_size=time_step_size,J=J,A=A,gamma=gamma)

        jac = lambda U: objective_jac(U.reshape(N, time_steps, dc),chis=chis,radar_states=ps,target_states=m0,time_step_size=time_step_size,J=J,A=A,gamma=gamma)

        constraint_t2s_fn = lambda U: collision_constraint(U.reshape(N,time_steps,dc),ps,m0_rollout)
        constraint_t2s = NonlinearConstraint(constraint_t2s_fn,R_sensors_to_targets,np.inf)
        con = [constraint_t2s]
        # constraint_t2s_fn = lambda U: collision_constraint(U.reshape(N,time_steps,dc),ps,m0_rollout) - R_sensors_to_targets
        # con = {'type':'ineq','fun':constraint_t2s_fn}

        opt_sol = minimize(fun=objective,x0=U.ravel(),method="trust-constr",bounds=bnds,constraints=con
                           ,options={"maxiter":1000,"disp":True})
        U = opt_sol.x.reshape(N,time_steps,dc)
        # obj_value = opt_sol.f
        _, _, Sensor_Positions, Sensor_Chis = state_multiple_update_parallel(ps,
                                                                                            U,
                                                                                            chis, time_step_size)
        ps = Sensor_Positions[:,1,:]
        chis = Sensor_Chis[:,1,:]

        # print(ps.shape,chis.shape,ps.squeeze().shape)
        # ps = ps.squeeze()
        # chis = chis.squeeze()


        end = time()
        print(f"Step {k} Optimization Time: ",end-start)

        # m0  = ps

        J = IM_fn(radar_states=ps,target_states=m0,J=J) #[JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],s=s) for i in range(len(Js))]

        # print([jnp.linalg.slogdet(Ji)[1].item() for Ji in Js])
        J_list.append(jnp.linalg.slogdet(J)[1].ravel())

        print("FIM (higher is better) ",np.sum(J_list[-1]))

        save_time = time()
        if (k+1)%frame_skip == 0:
            # fig_mpc.minorticks_off()

            # Target "Tracking"
            for m in range(M):
                axes_mpc[0].add_patch(
                    Circle(m0[m, :], radius_projected, edgecolor="green", fill=False, lw=1,
                           linestyle="--", label="_nolegend_"))

            axes_mpc[0].plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="_nolegend_")
            axes_mpc[0].plot(m0[:,0], m0[:,1], 'go',label="Targets")
            axes_mpc[0].plot(ps_init[:,0], ps_init[:,1], 'md',label="Sensor Init")
            axes_mpc[0].plot(ps[:,0], ps[:,1], 'rx',label="Sensors Next Position")
            axes_mpc[0].plot(Sensor_Positions[:,0,0], Sensor_Positions[:,0,1], 'r*',label="Sensor Position")
            axes_mpc[0].plot(Sensor_Positions[:,1:,0].T, Sensor_Positions[:,1:,1].T, 'r.-',label="_nolegend_")
            axes_mpc[0].plot([],[],"r.-",label="Sensor Planned Path")

            axes_mpc[0].legend(bbox_to_anchor=(0.7, 0.95))
            axes_mpc[0].set_title(f"k={k}")
            axes_mpc[0].axis('equal')

            qx,qy,logdet_grid = FIM_Visualization(ps=ps, qs=m0,
                                                  Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW,
                                                  N=1000)

            axes_mpc[1].contourf(qx, qy, logdet_grid, levels=20)
            axes_mpc[1].scatter(ps[:, 0], ps[:, 1], s=50, marker="x", color="r")
            #
            axes_mpc[1].scatter(m0[:, 0], m0[:, 1], s=50, marker="o", color="g")
            axes_mpc[1].set_title("Instant Time Objective Function Map")
            axes_mpc[1].axis('equal')

            axes_mpc[2].plot(jnp.array(J_list),"b-",label="Total FIM")
            # axes_mpc[2].plot(jnp.array(J_list),"r-",label="Individual FIM")
            axes_mpc[2].set_ylabel("Target logdet FIM (Higher is Better)")
            axes_mpc[2].set_title(f"Avg MPPI LogDet FIM={np.round(J_list[-1])}")
            axes_mpc[2].set_xlabel("Time Step")


            filename = f"frame_{k}.png"
            images_mpc.append(os.path.join(photo_dump, filename))

            fig_mpc.tight_layout()
            fig_mpc.savefig(os.path.join(photo_dump, filename))

            axes_mpc[0].cla()
            axes_mpc[1].cla()
            axes_mpc[2].cla()

            # record the controls over time ...
            file_control = os.path.join(photo_dump, f"scipy_control_{k}.png")
            images_control.append(file_control)
            for n in range(N):
                axes_control[0].plot(U[n, :, 0].T, '.-', color=colors[n], label="_nolegend_")
                axes_control[1].plot(U[n, :, 1].T, '.-', color=colors[n], label="_nolegend_")

            axes_control[0].set_title("Velocity [m/s]")
            axes_control[0].set_xlabel("Time Step")
            axes_control[1].set_title("Ang Velocity [rad/s]")
            axes_control[1].set_xlabel("Time Step")
            fig_control.suptitle(f"Step={k}")
            fig_control.tight_layout()
            fig_control.savefig(file_control)
            axes_control[0].cla()
            axes_control[1].cla()

        save_time = time() - save_time
        print("Figure Save Time: ",save_time)

        U = jnp.roll(U,-1,axis=1)


    plt.figure()
    plt.plot(jnp.array(J_list))
    plt.show()
    print("lol")

    frames = []
    for frame_name in images_mpc:
        frames.append(imageio.imread(frame_name))

    imageio.mimsave(os.path.join(gif_savepath, gif_mpc_filename), frames, duration=0.25)

    frames = []
    for frame_name in images_control:
        frames.append(imageio.imread(frame_name))

    imageio.mimsave(os.path.join(gif_savepath, gif_control_filename), frames, duration=0.25)

    if remove_photo_dump:
        for filename in images_mpc+images_control:
            os.remove(filename)