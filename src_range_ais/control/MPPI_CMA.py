from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jaxopt import ScipyMinimize



from src_range_ais.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,Single_FIM_Radar,Multi_FIM_Logdet_decorator_MPC,FIM_Visualization

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle


from tqdm import tqdm
from time import time

import numpy as np
import jax
import jax.numpy as jnp

from src_range_ais.utils import NoiseParams
from src_range_ais.control.MPPI import *
from copy import deepcopy

import os


if __name__ == "__main__":
    import matplotlib as mpl

    mpl.rcParams['path.simplify_threshold'] = 1.0
    import matplotlib.style as mplstyle

    mplstyle.use('fast')
    mplstyle.use(['ggplot', 'fast'])

    seed = 123
    key = jax.random.PRNGKey(seed)
    np.random.seed(123)

    gif_savepath = os.path.join("..", "..", "images", "gifs")
    os.makedirs("tmp_images",exist_ok=True)

    # Experiment Choice
    update_steps = 0

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = 0.1
    NT = 5
    MPPI_FLAG = True
    PRUNE_FLAG = False
    MPPI_VISUALIZE = False
    MPPI_ITER_VISUALIZE = True

    N = 6

    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e6;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    # alpha = (jnp.pi)**2 / 3
    B = 0.05 * 10**5

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000

    Pt = 10000
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    Pr = K / (R ** 4)

    # get the power of the noise of the signalf
    SNR=0


    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    time_steps = 15
    R_sensors_to_targets = 5.
    R_sensors_to_sensors = 1.5
    time_step_size = T
    max_velocity = 50.
    min_velocity = 0
    max_angle_velocity = jnp.pi
    min_angle_velocity = -jnp.pi

    # ==================== MPPI CONFIGURATION ================================= #
    limits = jnp.array([[max_velocity, max_angle_velocity], [min_velocity, min_angle_velocity]])
    stds = jnp.array([[-3,3],
                      [-30* jnp.pi/180, 30 * jnp.pi/180]])

    v_std = 15
    av_std = jnp.pi/180 * 45
    cov_timestep = jnp.array([[v_std**2,0],[0,av_std**2]])
    cov_traj = jax.scipy.linalg.block_diag(*[cov_timestep for t in range(time_steps)])
    # cov = jax.scipy.linalg.block_diag(*[cov_traj for n in range(N)])
    cov = jnp.stack([cov_traj for n in range(N)])
    v_init = 5
    av_init = jnp.pi/180 * 90
    mu = jnp.tile(jnp.array([v_init,av_init]),(N*time_steps,))

    num_traj = 300
    MPPI_iterations = 50
    MPPI_method = "single"
    method = "Single_FIM_3D_action_MPPI"
    u_ptb_method = "mixture"
    fim_method = "Standard FIM"
    # ==================== AIS CONFIGURATION ================================= #
    temperature = 0.1
    alpha = 0.95
    elite_threshold = 0.8
    AIS_method = "information"

    gif_savename =  f"AIS_MPPI_{AIS_method}.gif"


    from copy import deepcopy
    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100)

    ps_init = deepcopy(ps)
    z_elevation = 10
    # qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
    #                 [-50.4,30.32,z_elevation,-20,-10,0], #,
    #                 [10,10,z_elevation,10,10,0],
    #                 [20,20,z_elevation,5,-5,0]])
    qs = jnp.array([[0.0, -0.0,z_elevation, 0., 0,0], #,#,
                    [-50.4,30.32,z_elevation,-0,-0,0], #,
                    [10,10,z_elevation,0,0,0],
                    [20,20,z_elevation,0,0,0]])

    M, dm = qs.shape;
    N , dn = ps.shape;

    sigmaW = jnp.sqrt(M*Pr/ (10**(SNR/10)))
    # coef = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi)** 3 / (R ** 4)
    C = c**2 * sigmaW**2 / (jnp.pi**2 * 8 * fc**2) * 1/K

    print("Noise Power: ",sigmaW**2)
    print("Power Return (RCS): ",Pr)
    print("K",K)

    print("Pt (peak power)={:.9f}".format(Pt))
    print("lam ={:.9f}".format(lam))
    print("C=",C)

    # ============================ MPC Settings =====================================#
    gamma = 0.95
    paretos = jnp.ones((M,)) * 1 / M  # jnp.array([1/3,1/3,1/3])
    assert len(paretos) == M, "Pareto weights not equal to number of targets!"
    assert (jnp.sum(paretos) <= (1 + 1e-5)) and (jnp.sum(paretos) >= -1e-5), "Pareto weights don't sum to 1!"

    sigmaQ = np.sqrt(10 ** -1)
    sigmaV = jnp.sqrt(1)

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
    Q = jnp.kron(jnp.eye(M), Q_single);
    G = jnp.eye(N)

    nx = Q.shape[0]

    J = jnp.eye(dm*M) #jnp.stack([jnp.eye(d) for m in range(M)])


    # IM_fn = partial(Single_FIM_Radar,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW)
    # IM_fn(ps,qs[[0],:],Js=Js)
    Qinv = jnp.linalg.inv(Q+jnp.eye(dm*M)*1e-8)

    if fim_method == "PCRLB":
        IM_fn = partial(Single_JU_FIM_Radar, A=A, Qinv=Qinv, Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, fc=fc, c=c,
                        sigmaV=sigmaV, sigmaW=sigmaW)
    elif fim_method == "Standard FIM":
        IM_fn = partial(Single_FIM_Radar,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW)

    # IM_fn_parallel = vmap(IM_fn, in_axes=(None, 0, 0))

    Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC(IM_fn=IM_fn,method=method)

    MPPI_scores = MPPI_scores_wrapper(Multi_FIM_Logdet,method=MPPI_method)

    if AIS_method == "CE":
        weight_fn = partial(weighting(AIS_method),elite_threshold=elite_threshold)
    elif AIS_method == "information":
        weight_fn = partial(weighting(AIS_method),temperature=temperature)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    # time_step_sizes = jnp.tile(time_step_size, (N, 1))

    state_multiple_update_vmap = vmap(unicycle_kinematics, (0, 0, 0, None))

    m0 = qs
    images = []; images_mppi = []; images_mppi_control = []; images_costs = []
    FIMs = []
    qs_previous = None

    U_V = jnp.ones((N,time_steps,1)) * v_init
    U_W = jnp.ones((N,time_steps,1)) * av_init
    U =jnp.concatenate((U_V,U_W),axis=-1)
    mu_bias = jnp.zeros_like(U)
    mu_bias = mu_bias.at[:,:,0].set(1)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    fig_debug,axes_debug = plt.subplots(1,1,figsize=(3,3))

    fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))

    fig_costs,axes_costs = plt.subplots(1,2,figsize=(10,5))

    for k in range(NT):
        print(f"\n Step {k} MPPI Iteration: ")
        qs_previous = m0
        m0 = (A @ m0.reshape(-1, 1)).reshape(M, dm)

        best_mppi_iter_score = np.inf
        mppi_round_time_start = time()

        target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])

        U_prime = deepcopy(U)
        cov_prime = deepcopy(cov)

        for mppi_iter in range(MPPI_iterations):
            start = time()
            key, subkey = jax.random.split(key)

            mppi_start = time()

            E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).reshape(N,-1), cov=cov_prime, shape=(num_traj,N),method="svd")

            # simulate the model with the trajectory noise samples
            V = U_prime + E.reshape(num_traj,N,time_steps,2)

            mppi_rollout_start = time()

            P_MPPI,CHI_MPPI = MPPI_CMA(U_MPPI=V, chis_nominal=chis,
                                                               ps=ps,
                                                               time_step_size=time_step_size)
            mppi_rollout_end = time()


            # Score all the rollouts
            mppi_score_start = time()
            trajectory_cost = MPPI_scores(ps, target_states=target_states_rollout, U_MPPI=V,chis=chis, time_step_size=time_step_size,
                                      A=A,J=J,
                                      gamma=gamma)

            mppi_score_end = time()

            # control_cost = temperature * (1-alpha) * V.reshape(num_traj,N,1,-1) @ jnp.linalg.inv(cov) @ V.reshape(num_traj,N,-1,1)
            control_cost =  U_prime.reshape(N,1,-1) @ jnp.linalg.inv(cov) @ (V-U).reshape(num_traj,N,-1,1)

            total_cost = trajectory_cost.reshape(num_traj,1) + temperature * (1-alpha) * control_cost.reshape(num_traj,N)

            min_idx = jnp.argmin(total_cost)
            lowest_cost = total_cost[min_idx]


            weights = weight_fn(total_cost)
            neff = 1/(jnp.sum(weights**2))

            if neff < N*time_steps:
                print("Weight Tempering")
                weights = weights ** (1 / 2)
                weights = weights / jnp.sum(weights,0)

                neff = 1 / (jnp.sum(weights ** 2))

            if jnp.isnan(neff).any():
                print("BREAK!")

            print(f"Neff Samples {int(neff)}")

            # delta_actions = U_MPPI - U
            # U = jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1), axis=0)

            if mppi_iter < (MPPI_iterations-1):
                # U += jnp.sum(delta_actions * scores_MPPI_weight.reshape(-1, 1, 1, 1), axis=0)

                U_prime = U_prime + jnp.sum(weights.reshape(num_traj,N,1,1) * E.reshape(num_traj,N,time_steps,2),axis=0)

                # diff = U_MPPI.reshape(num_traj,N,-1) - U.reshape(N,-1)

                cov_prime = jnp.sum(weights.reshape(num_traj,N,1,1) *(E[:,:,:,jnp.newaxis] @ E[:,:,jnp.newaxis,:]),axis=0) + jnp.tile(jnp.eye(cov.shape[-1]),(N,1,1))*1e-8



            if k==0 and MPPI_ITER_VISUALIZE:
                file_mppi = os.path.join("tmp_images", f"MPPI_single_iteration_{mppi_iter}.png")
                images_mppi.append(file_mppi)
                axes_debug.plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="Target Init Position")
                axes_debug.plot(m0[:,0], m0[:,1], 'go',label="Target Position")
                _, _, Sensor_Positions_debug, Sensor_Chis_debug = vmap(unicycle_kinematics, (0, 0, 0, None))(jnp.expand_dims(ps, 1), U_prime,chis, time_step_size)
                _, _, Sensor_Positions_best_debug, Sensor_Chis_best_debug = vmap(unicycle_kinematics, (0, 0, 0, None))(jnp.expand_dims(ps, 1), U,chis, time_step_size)

                if MPPI_VISUALIZE:
                    for n in range(N):
                        plt.plot(P_MPPI[:, n, :, 0].T, P_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
                axes_debug.plot(Sensor_Positions_debug[:,0,0], Sensor_Positions_debug[:,0,1], 'r*',label="Sensor Position")
                axes_debug.plot(Sensor_Positions_debug[:,1:,0].T, Sensor_Positions_debug[:,1:,1].T, 'r-',label="_nolegend_")
                axes_debug.plot(Sensor_Positions_best_debug[:,1:,0].T, Sensor_Positions_best_debug[:,1:,1].T, color="lime",linestyle='-',label="_nolegend_")

                fig_debug.tight_layout()
                fig_debug.savefig(file_mppi)
                axes_debug.cla()

                # Control Input
                file_control= os.path.join("tmp_images", f"MPPI_control_single_iteration_{mppi_iter}.png")
                images_mppi_control.append(file_control)
                axes_control[0].plot(U_prime[:,:,0].T, 'g.-',label="Velocity Input")
                axes_control[0].set_title("Velocity [m/s]")
                axes_control[0].set_xlabel("Time Step")
                axes_control[1].plot(U_prime[:,:,1].T, 'm.-',label="Ang Velocity Input")
                axes_control[1].set_title("Ang Velocity [rad/s]")
                axes_control[1].set_xlabel("Time Step")

                fig_control.tight_layout()
                fig_control.savefig(file_control)
                axes_control[0].cla()
                axes_control[1].cla()

                # Costs
                file_cost= os.path.join("tmp_images", f"MPPI_cost_single_iteration_{mppi_iter}.png")
                images_costs.append(file_cost)
                axes_costs[0].bar(np.arange(N),jnp.sum(weights*trajectory_cost.reshape(-1,1),axis=0),color='g',label="Objective Cost")
                axes_costs[0].set_title("Trajectory Cost")
                axes_costs[0].set_xlabel("Radar j")
                axes_costs[1].bar(np.arange(N),jnp.sum(control_cost.squeeze() * weights,axis=0), color='m',label="Control Cost")
                axes_costs[1].set_title("Control Cost")
                axes_costs[1].set_xlabel("Radar j")

                fig_costs.tight_layout()
                fig_costs.savefig(file_cost)
                axes_costs[0].cla()
                axes_costs[1].cla()
            # U_BEST = jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
            mppi_end = time()
            # print("MPPI Mean Score: ",np.mean(scores_MPPI))
            # print("MPPI Best Score: ",np.argmin(scores_MPPI))
            # print("MPPI TIME: ",mppi_end-mppi_start)

        mppi_round_time_end = time()
        mean_shift = (U_prime - U)

        E_prime = E + mean_shift.reshape(1,N,time_steps*2)

        U = U + jnp.sum(E_prime.reshape(num_traj,N,time_steps,2),axis=0)
        # U +=
        print("MPPI Round Time: ",mppi_round_time_end-mppi_round_time_start)
        print("MPPI Iter Time: ",mppi_end-mppi_start)
        print("MPPI Score Time: ",mppi_score_end-mppi_score_start)
        print("MPPI Mean Score: ",jnp.nanmean(total_cost))
        print("MPPI Best Score: ",-best_mppi_iter_score)
        # FIMs.append(-jnp.nanmean(scores_MPPI))



        # U_BEST =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
        # U_nominal =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
        _, _, Sensor_Positions, Sensor_Chis = state_multiple_update_vmap(jnp.expand_dims(ps, 1), U ,
                                                                       chis, time_step_size)

        # U += jnp.clip(jnp.sum(weights.reshape(num_traj,1,1,1) *  E.reshape(num_traj,N,time_steps,2),axis=0),U_lower,U_upper)
        U = jnp.roll(U,-1,axis=1)

        # if k == 0:
        #     MPPI_visualize(P_MPPI, Sensor_Positions)
        # print(ps.shape,chis.shape,ps.squeeze().shape)
        ps = Sensor_Positions[:,1,:]
        chis = Sensor_Chis[:,1]
        Sensor_Positions = np.asarray(Sensor_Positions)

        J = IM_fn(radar_states=ps,target_states=m0,J=J) #[JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],s=s) for i in range(len(Js))]

         # print(jnp.trace(J))
        FIMs.append(jnp.linalg.slogdet(J)[1].ravel())
        file = os.path.join("tmp_images",f"JU_test_target_movement{k}.png")
        if ((k+1) % frame_skip) == 0:
            fig_time = time()
            # for n in range(N):
            #     axes[0].plot(P_MPPI[:,n,:,0].T,P_MPPI[:,n,:,1].T,'b-',label="__nolegend__")
            axes[0].plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="Target Init Position")
            axes[0].plot(m0[:,0], m0[:,1], 'go',label="Target Position")
            axes[0].plot(ps_init[:,0], ps_init[:,1], 'md',label="Sensor Init")


            if MPPI_VISUALIZE:
                for n in range(N):
                    axes[0].plot(P_MPPI[:, n, :, 0].T, P_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
            axes[0].plot(Sensor_Positions.squeeze()[:,0,0], Sensor_Positions.squeeze()[:,0,1], 'r*',label="Sensor Position")
            axes[0].plot(Sensor_Positions.squeeze()[:,1:,0].T, Sensor_Positions.squeeze()[:,1:,1].T, 'r-',label="_nolegend_")
            axes[0].plot([],[],"r.-",label="Sensor Planned Path")
            axes[0].set_title(f"k={k}")
            # axes[0].legend(bbox_to_anchor=(0.5, 1.45),loc="upper center")
            axes[0].legend(bbox_to_anchor=(0.7, 1.45),loc="upper center")

            qx,qy,logdet_grid = FIM_Visualization(ps=ps, qs=m0,
                                                  Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW,
                                                  N=1000)

            axes[1].contourf(qx, qy, logdet_grid, levels=20)
            axes[1].scatter(ps[:, 0], ps[:, 1], s=50, marker="x", color="r")
            #
            axes[1].scatter(m0[:, 0], m0[:, 1], s=50, marker="o", color="g")
            axes[1].set_title("Instant Time Objective Function Map")

            axes[2].plot(jnp.array(FIMs),'ko')
            axes[2].set_ylabel("LogDet FIM (Higher is Better)")
            axes[2].set_title(f"Avg MPPI LogDet FIM={np.round(FIMs[-1])}")
            fig.tight_layout()
            fig.savefig(file)

            axes[0].cla()
            axes[1].cla()
            axes[2].cla()
            fig_time = time()-fig_time
            images.append(file)

        print("Fig Save Time: ",fig_time)
    images = [imageio.imread(file) for file in images]
    imageio.mimsave(os.path.join(gif_savepath,gif_savename),images,duration=0.1)#                              f"../../images/gifs/FIM_Kalman/JU_test_sensor_and_target.gif",images,duration=.1)


    if MPPI_ITER_VISUALIZE:
        images = [imageio.imread(file) for file in images_mppi]
        imageio.mimsave(os.path.join(gif_savepath,'MPPI_step.gif'),images,duration=0.1)#

    images = [imageio.imread(file) for file in images_mppi_control]
    imageio.mimsave(os.path.join(gif_savepath, 'MPPI_control_step.gif'), images, duration=0.1)

    images = [imageio.imread(file) for file in images_costs]
    imageio.mimsave(os.path.join(gif_savepath, 'MPPI_cost_step.gif'), images, duration=0.1)  #
    #