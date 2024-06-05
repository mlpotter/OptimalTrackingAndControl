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

from sklearn.covariance import OAS

from tqdm import tqdm
from time import time

import numpy as np
import jax
import jax.numpy as jnp

from src_range_ais.utils import NoiseParams
from src_range_ais.control.MPPI import *
from src_range_ais.objective_fns.objectives import *
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
    tmp_img_savepath = os.path.join("tmp_images")
    img_savepath = os.path.join("..","..","images")
    os.makedirs(tmp_img_savepath,exist_ok=True)

    # Experiment Choice
    update_steps = 0

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = 0.1
    NT = 115
    MPPI_FLAG = True
    PRUNE_FLAG = False
    MPPI_VISUALIZE = False
    MPPI_ITER_VISUALIZE = True

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
    time_step_size = T
    max_velocity = 50.
    min_velocity = 0
    max_angle_velocity = jnp.pi
    min_angle_velocity = -jnp.pi

    # ==================== MPPI CONFIGURATION ================================= #
    v_std = 5
    av_std = jnp.pi/180 * 45
    cov_timestep = jnp.array([[v_std**2,0],[0,av_std**2]])
    cov_traj = jax.scipy.linalg.block_diag(*[cov_timestep for t in range(time_steps)])
    cov = jax.scipy.linalg.block_diag(*[cov_traj for n in range(N)])
    # cov = jnp.stack([cov_traj for n in range(N)])
    v_init = 50
    av_init = 180 * jnp.pi/180
    mu = jnp.tile(jnp.array([v_init,av_init]),(N*time_steps,))

    num_traj = 200
    MPPI_iterations = 50
    MPPI_method = "single"
    method = "Single_FIM_3D_action_MPPI"
    u_ptb_method = "normal"
    fim_method = "Standard FIM"
    # ==================== AIS CONFIGURATION ================================= #
    temperature = 0.1
    elite_threshold = 0.95
    AIS_method = "CE"

    from copy import deepcopy
    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100)

    ps_init = deepcopy(ps)
    z_elevation = 10
    # qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
    #                 [-50.4,30.32,z_elevation,-20,-10,0], #,
    #                 # [10,10,z_elevation,10,10,0],
    #                 [20,20,z_elevation,5,-5,0]])
    # qs = jnp.array([[0.0, -0.0,z_elevation, 0., 0,0], #,#,
    #                 [-50.4,30.32,z_elevation,-0,-0,0], #,
    #                 [10,10,z_elevation,0,0,0],
    #                 [20,20,z_elevation,0,0,0]])
    qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
                    [-100.4,-30.32,z_elevation,20,-10,0], #,
                    [30,30,z_elevation,-10,-10,0]])#,

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
    # spread = 15 alpha=100 works
    spread_target = 20
    spread_radar = 2
    speed_minimum = 5
    R_sensors_to_targets = 25
    R_sensors_to_sensors = 10

    alpha1 = 1 # FIM
    alpha2 = 80 # Target - Radar Distance
    alpha3 = 60 # Radar - Radar Distance
    alpha4 = 1 # AIS control cost
    alpha5 = 0 # speed cost

    theta = jnp.arcsin(z_elevation/R_sensors_to_targets)
    radius_projected = R_sensors_to_targets * jnp.cos(theta)

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

    # weight_info =partial(weighting("information"),temperature=temperature)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    # time_step_sizes = jnp.tile(time_step_size, (N, 1))

    state_multiple_update_vmap = vmap(unicycle_kinematics, (0, 0, 0, None))

    collision_penalty_vmap = jit( vmap(collision_penalty, in_axes=(0, None, None, None)))
    self_collision_penalty_vmap = jit(vmap(self_collision_penalty, in_axes=(0, None, None)))
    speed_penalty_vmap = jit(vmap(speed_penalty, in_axes=(0, None)))

    m0 = qs

    FIMs = []
    qs_previous = None

    U_V = jnp.ones((N,time_steps,1)) * v_init
    U_W = jnp.ones((N,time_steps,1)) * av_init
    U =jnp.concatenate((U_V,U_W),axis=-1)
    mu_bias = jnp.zeros_like(U)
    mu_bias = mu_bias.at[:,:,0].set(1)

    fig_mppi_debug,axes_mppi_debug = plt.subplots(1,1,figsize=(10,10))
    fig_control_debug,axes_control_debug = plt.subplots(1,2,figsize=(10,5))
    fig_costs_debug,axes_costs_debug = plt.subplots(1,4,figsize=(20,5))


    fig_main,axes_main = plt.subplots(1,3,figsize=(15,5))
    fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))
    fig_costs,axes_costs = plt.subplots(1,4,figsize=(20,5))
    fig_neff,axes_neff = plt.subplots(1,2,figsize=(10,5))

    cost_means = np.array([None]*NT)
    cost_stds = np.array([None]*NT)
    neffs  = np.array([None]*NT)

    cost_means_debug = np.array([None]*NT)
    cost_stds_debug = np.array([None]*NT)
    neffs_debug  = np.array([None]*MPPI_iterations)

    images_main = [];
    images_control = [];
    images_costs = []
    images_mppi_debug = []
    images_control_debug = [];
    images_costs_debug = [];

    for k in range(NT):
        print(f"\n Step {k} MPPI Iteration: ")
        qs_previous = m0
        m0 = (A @ m0.reshape(-1, 1)).reshape(M, dm)

        best_mppi_iter_score = np.inf
        mppi_round_time_start = time()

        target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])

        U_prime = deepcopy(U)
        cov_prime = deepcopy(cov)
        best_cost = jnp.inf
        for mppi_iter in range(MPPI_iterations):
            start = time()
            key, subkey = jax.random.split(key)

            mppi_start = time()

            E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime, shape=(num_traj,),method="svd")

            # simulate the model with the trajectory noise samples
            V = U_prime + E.reshape(num_traj,N,time_steps,2)

            mppi_rollout_start = time()

            P_MPPI,CHI_MPPI = MPPI_CMA(U_MPPI=V, chis_nominal=chis,
                                                               ps=ps,
                                                               time_step_size=time_step_size)
            mppi_rollout_end = time()


            # Score all the rollouts
            mppi_score_start = time()
            cost_trajectory = MPPI_scores(ps, target_states=target_states_rollout, U_MPPI=V,chis=chis, time_step_size=time_step_size,
                                      A=A,J=J,
                                      gamma=gamma)

            mppi_score_end = time()

            # control_cost = (V-U).reshape(num_traj,1,-1) @ jnp.linalg.inv(cov) @ (V-U).reshape(num_traj,-1,1)
            cost_collision_r2t = jnp.sum(jnp.column_stack([collision_penalty_vmap(P_MPPI[:,:,t],target_states_rollout[t-1],R_sensors_to_targets,spread_target) for t in range(1,time_steps+1)]) * (gamma**(jnp.arange(time_steps))) / jnp.sum(gamma**(jnp.arange(time_steps))) , axis=-1)

            cost_collision_r2r = jnp.sum(jnp.column_stack([self_collision_penalty_vmap(P_MPPI[:,:,t],R_sensors_to_sensors,spread_radar) for t in range(1,time_steps+1)]) * (gamma**(jnp.arange(time_steps))) / jnp.sum(gamma**(jnp.arange(time_steps))) , axis=-1)

            cost_control = ((U_prime - U).reshape(1,1,-1) @ jnp.linalg.inv(cov) @ (V).reshape(num_traj, -1,1)).ravel()
            # cost_control = control_penalty(U_prime=U_prime,U=U,V=V,cov=cov,dTraj=num_traj,dN=N,dC=2)

            cost_speed = jnp.sum(jnp.column_stack([speed_penalty_vmap(V[:,:,t],speed_minimum) for t in range(time_steps)]) * (gamma**(jnp.arange(time_steps))) / jnp.sum(gamma**(jnp.arange(time_steps))) , axis=-1)

            # print(speed_cost.mean(),speed_cost.mean()*alpha2)

            cost_MPPI = alpha1*cost_trajectory + alpha2*cost_collision_r2t + alpha3 * cost_collision_r2r * temperature * (1-alpha4) * cost_control + alpha5*cost_speed



            min_idx = jnp.argmin(cost_MPPI)
            lowest_cost = cost_MPPI[min_idx]



            weights = weight_fn(cost_MPPI)

            neff = 1/(jnp.sum(weights**2))

            # if neff < N*time_steps:
            #     print("Weight Tempering")f
            #     weights = weights ** (1 / 2)
            #     weights = weights / jnp.sum(weights,0)
            #
            #     neff = 1 / (jnp.sum(weights ** 2))

            if jnp.isnan(cost_MPPI).any():
                print("BREAK!")
                break

            # print(f"Neff Samples {int(neff)}")

            # delta_actions = U_MPPI - U
            # U = jnp.sum(U_MPPI * weights.reshape(-1, 1, 1, 1), axis=0)
            if (mppi_iter < (MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):
                # U += jnp.sum(delta_actions * weights.reshape(-1, 1, 1, 1), axis=0)
                best_cost = jnp.sum(cost_MPPI*weights)
                neffs_debug[mppi_iter] = 1/ (jnp.sum(weights**2))

                if k==0:
                    print(f"Best Cost @ {mppi_iter} : {best_cost}, Speed Cost @ {alpha5*cost_speed.mean()}")

                U_copy = deepcopy(U_prime)
                U_prime = U_prime + jnp.sum(weights.reshape(num_traj,1,1,1) * E.reshape(num_traj,N,time_steps,2),axis=0)

                # diff = U_MPPI.reshape(num_traj,N,-1) - U.reshape(N,-1)

                # cov_prime = jnp.sum(weights.reshape(-1,1,1) *(E[:,:,jnp.newaxis] @ E[:,jnp.newaxis,:]),axis=0) + jnp.eye(cov.shape[-1])*1e-4
                oas = OAS(assume_centered=True).fit(E[weights != 0])
                cov_prime = jnp.array(oas.covariance_)
            #
            if k==0 and MPPI_ITER_VISUALIZE:
                neffs_debug[mppi_iter] = 1/ (jnp.sum(weights**2))
                # save subiterations of MPPI
                file_mppi_debug = os.path.join(tmp_img_savepath, f"MPPI_subiteration_{mppi_iter}.png")
                images_mppi_debug.append(file_mppi_debug)
                axes_mppi_debug.plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="Target Init Position")
                axes_mppi_debug.plot(m0[:,0], m0[:,1], 'go',label="Target Position")
                _, _, Sensor_Positions_debug, Sensor_Chis_debug = vmap(unicycle_kinematics, (0, 0, 0, None))(jnp.expand_dims(ps, 1), U,chis, time_step_size)
                _, _, Sensor_Positions_best_debug, Sensor_Chis_best_debug = vmap(unicycle_kinematics, (0, 0, 0, None))(jnp.expand_dims(ps, 1), U_prime,chis, time_step_size)

                if MPPI_VISUALIZE:
                    for n in range(N):
                        plt.plot(P_MPPI[:, n, :, 0].T, P_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
                axes_mppi_debug.plot(Sensor_Positions_debug[:,0,0], Sensor_Positions_debug[:,0,1], 'r*',label="Sensor Position")
                axes_mppi_debug.plot(Sensor_Positions_debug[:,1:,0].T, Sensor_Positions_debug[:,1:,1].T, 'r-',label="_nolegend_")
                axes_mppi_debug.plot(Sensor_Positions_best_debug[:,1:,0].T, Sensor_Positions_best_debug[:,1:,1].T, color="lime",linestyle='-',label="_nolegend_")

                fig_mppi_debug.suptitle(f"MPPI Subiteration {mppi_iter}")
                fig_mppi_debug.tight_layout()
                fig_mppi_debug.savefig(file_mppi_debug)
                axes_mppi_debug.cla()

                # save subiterations of control input
                file_control_debug= os.path.join(tmp_img_savepath, f"MPPI_control_subiteration_{mppi_iter}.png")
                images_control_debug.append(file_control_debug)
                for n in range(N):
                    axes_control_debug[0].plot(U_prime[n, :, 0].T, '.-',color=colors[n],label="_nolegend_")
                    axes_control_debug[1].plot(U_prime[n, :, 1].T, '.-', color=colors[n], label="_nolegend_")

                axes_control_debug[0].set_title("Velocity [m/s]")
                axes_control_debug[0].set_xlabel("Time Step")
                axes_control_debug[1].set_title("Ang Velocity [rad/s]")
                axes_control_debug[1].set_xlabel("Time Step")

                fig_control_debug.suptitle(f"MPPI Subiteration {mppi_iter}")
                fig_control_debug.tight_layout()
                fig_control_debug.savefig(file_control_debug)
                axes_control_debug[0].cla()
                axes_control_debug[1].cla()

                # save subiterations of costs
                file_costs_debug= os.path.join(tmp_img_savepath, f"MPPI_costs_subiteration_{mppi_iter}.png")
                images_costs_debug.append(file_costs_debug)

                axes_costs_debug[0].stem(cost_trajectory*alpha1)
                axes_costs_debug[1].stem(cost_collision_r2t*alpha2)
                axes_costs_debug[2].stem(cost_collision_r2r*alpha3)
                axes_costs_debug[3].stem(cost_speed*alpha5)

                axes_costs_debug[0].set_title(f"State , $\\alpha_1$={alpha1}")
                axes_costs_debug[1].set_title(f"Collision R2T , $\\alpha_2$={alpha2}")
                axes_costs_debug[2].set_title(f"Collision R2R , $\\alpha_3$={alpha3}")
                axes_costs_debug[3].set_title(f"Speed , $\\alpha_5$={alpha5}")

                [ax.set_xlabel("Rollout") for ax in axes_costs_debug]


                fig_costs_debug.suptitle(f"MPPI Subiteration {mppi_iter}")
                fig_costs_debug.tight_layout()
                fig_costs_debug.savefig(file_costs_debug)
                [ax.cla() for ax in axes_costs_debug]

            # U_BEST = jnp.sum(U_MPPI * weights.reshape(-1, 1, 1, 1),axis=0)
            mppi_end = time()
            # print("MPPI Mean Score: ",np.mean(scores_MPPI))
            # print("MPPI Best Score: ",np.argmin(scores_MPPI))
            # print("MPPI TIME: ",mppi_end-mppi_start)

        mppi_round_time_end = time()

        if jnp.isnan(cost_MPPI).any():
            print("BREAK!")
            break

        weights = weight_fn(cost_MPPI)

        cost_means[k] = cost_MPPI.mean()
        cost_stds[k] = cost_MPPI.std()
        neffs[k] = 1 / (jnp.sum(weights ** 2))

        mean_shift = (U_prime - U)

        E_prime = E + mean_shift.ravel()

        U += jnp.sum(weights.reshape(-1,1,1,1) * E_prime.reshape(num_traj,N,time_steps,2),axis=0)

        U = jnp.stack((jnp.clip(U[:,:,0],min_velocity,max_velocity),jnp.clip(U[:,:,1],min_angle_velocity,max_angle_velocity)),axis=-1)

        # U +=
        print("MPPI Round Time: ",mppi_round_time_end-mppi_round_time_start)
        print("MPPI Iter Time: ",mppi_end-mppi_start)
        print("MPPI Score Time: ",mppi_score_end-mppi_score_start)
        print("MPPI Mean Score: ",jnp.nanmean(cost_MPPI))
        print("MPPI Best Score: ",-best_mppi_iter_score)
        # FIMs.append(-jnp.nanmean(scores_MPPI))



        # U_BEST =  jnp.sum(U_MPPI * weights.reshape(-1, 1, 1, 1),axis=0)
        # U_nominal =  jnp.sum(U_MPPI * weights.reshape(-1, 1, 1, 1),axis=0)
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
        file = os.path.join(tmp_img_savepath,f"JU_test_target_movement{k}.png")
        if ((k+1) % frame_skip) == 0:
            fig_time = time()
            # for n in range(N):
            #     axes[0].plot(P_MPPI[:,n,:,0].T,P_MPPI[:,n,:,1].T,'b-',label="__nolegend__")

            # Main figure
            file_main = os.path.join(tmp_img_savepath, f"JU_test_target_movement{k}.png")
            images_main.append(file_main)
            axes_main[0].plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="Target Init Position")
            axes_main[0].plot(m0[:,0], m0[:,1], 'go',label="Target Position")
            axes_main[0].plot(ps_init[:,0], ps_init[:,1], 'md',label="Sensor Init")

            for m in range(M):
                axes_main[0].add_patch(
                    Circle(m0[m, :], radius_projected, edgecolor="green", fill=False, lw=1,
                           linestyle="--", label="_nolegend_"))

            for n in range(N):
                axes_main[0].add_patch(
                    Circle(Sensor_Positions[n,1, :], R_sensors_to_sensors, edgecolor="red", fill=False, lw=1,
                           linestyle="--", label="_nolegend_"))

            if MPPI_VISUALIZE:
                for n in range(N):
                    axes_main[0].plot(P_MPPI[:, n, :, 0].T, P_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
            axes_main[0].plot(Sensor_Positions.squeeze()[:,0,0], Sensor_Positions.squeeze()[:,0,1], 'r*',label="Sensor Position")
            axes_main[0].plot(Sensor_Positions.squeeze()[:,1:,0].T, Sensor_Positions.squeeze()[:,1:,1].T, 'r-',label="_nolegend_")
            axes_main[0].plot([],[],"r.-",label="Sensor Planned Path")
            axes_main[0].set_title(f"k={k}")
            # axes[0].legend(bbox_to_anchor=(0.5, 1.45),loc="upper center")
            axes_main[0].legend(bbox_to_anchor=(0.7, 1.45),loc="upper center")
            axes_main[0].axis('equal')

            qx,qy,logdet_grid = FIM_Visualization(ps=ps, qs=m0,
                                                  Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW,
                                                  N=1000)

            axes_main[1].contourf(qx, qy, logdet_grid, levels=20)
            axes_main[1].scatter(ps[:, 0], ps[:, 1], s=50, marker="x", color="r")
            #
            axes_main[1].scatter(m0[:, 0], m0[:, 1], s=50, marker="o", color="g")
            axes_main[1].set_title("Instant Time Objective Function Map")
            axes_main[1].axis('equal')

            axes_main[2].plot(jnp.array(FIMs),'ko')
            axes_main[2].set_ylabel("LogDet FIM (Higher is Better)")
            axes_main[2].set_title(f"Avg MPPI FIM={np.round(FIMs[-1])}")

            fig_main.suptitle(f"Iteration {k}")
            fig_main.tight_layout()
            fig_main.savefig(file_main)

            axes_main[0].cla()
            axes_main[1].cla()
            axes_main[2].cla()

            # Control Input
            file_control= os.path.join(tmp_img_savepath, f"MPPI_control_{k}.png")
            images_control.append(file_control)
            for n in range(N):
                axes_control[0].plot(U[n, :, 0].T, '.-', color=colors[n], label="_nolegend_")
                axes_control[1].plot(U[n, :, 1].T, '.-', color=colors[n], label="_nolegend_")

            axes_control[0].set_title("Velocity [m/s]")
            axes_control[0].set_xlabel("Time Step")
            axes_control[1].set_title("Ang Velocity [rad/s]")
            axes_control[1].set_xlabel("Time Step")

            fig_control.suptitle(f"Iteration {k}")
            fig_control.tight_layout()
            fig_control.savefig(file_control)
            axes_control[0].cla()
            axes_control[1].cla()


            # Costs
            file_control= os.path.join(tmp_img_savepath, f"MPPI_cost_{k}.png")
            images_costs.append(file_control)

            axes_costs[0].stem(cost_trajectory * alpha1)
            axes_costs[1].stem(cost_collision_r2t * alpha2)
            axes_costs[2].stem(cost_collision_r2r * alpha3)
            axes_costs[3].stem(cost_speed * alpha5)

            axes_costs[0].set_title(f"State , $\\alpha_1$={alpha1}")
            axes_costs[1].set_title(f"Collision R2T , $\\alpha_2$={alpha2}")
            axes_costs[2].set_title(f"Collision R2R , $\\alpha_3$={alpha3}")
            axes_costs[3].set_title(f"Speed , $\\alpha_5$={alpha5}")

            [ax.set_xlabel("Rollout") for ax in axes_costs]

            fig_costs.suptitle(f"MPPI Iter {k}")
            fig_costs.tight_layout()
            fig_costs.savefig(file_control)
            [ax.cla() for ax in axes_costs]



            fig_time = time()-fig_time

        print("Fig Save Time: ",fig_time)

    images = [imageio.imread(file) for file in images_main]
    imageio.mimsave(os.path.join(gif_savepath,f'MPPI_main_{AIS_method}_ais.gif'),images,duration=0.1)#

    images = [imageio.imread(file) for file in images_control]
    imageio.mimsave(os.path.join(gif_savepath, f'MPPI_control_{AIS_method}_ais.gif'), images, duration=0.1)

    images = [imageio.imread(file) for file in images_costs]
    imageio.mimsave(os.path.join(gif_savepath, f'MPPI_cost_{AIS_method}_ais.gif'), images, duration=0.1)  #

    images = [imageio.imread(file) for file in images_mppi_debug]
    imageio.mimsave(os.path.join(gif_savepath,f'MPPI_mppi_debug_{AIS_method}_ais.gif'),images,duration=0.1)#

    images = [imageio.imread(file) for file in images_control_debug]
    imageio.mimsave(os.path.join(gif_savepath, f'MPPI_control_debug_{AIS_method}_ais.gif'), images, duration=0.1)

    images = [imageio.imread(file) for file in images_costs_debug]
    imageio.mimsave(os.path.join(gif_savepath, f'MPPI_cost_debug_{AIS_method}_ais.gif'), images, duration=0.1)  #
    #
    for filename in images_main + images_control + images_costs + images_mppi_debug + images_control_debug + images_costs_debug:
        os.remove(filename)

    fig_misc,axes_misc = plt.subplots(1,3,figsize=(10,5))
    axes_misc[0].plot(cost_means*1/temperature,'ro-')
    axes_misc[0].set_title("Mean Trajectory Cost")
    axes_misc[1].plot(cost_stds*1/(temperature**2),'ro-')
    axes_misc[1].set_title("Std Trajectory Cost")
    fig_misc.suptitle(f"Temperture={temperature}")
    fig_misc.tight_layout()
    fig_misc.savefig(os.path.join(img_savepath,"MPPI_AIS_cost_analysis.png"))


    fig_neff,axes_neff = plt.subplots(1,2,figsize=(10,5))
    axes_neff[0].plot(neffs_debug,'ro-')
    axes_neff[0].set_title("Neffs Subiteration")
    axes_neff[0].set_xlabel("Subiteration")
    axes_neff[1].plot(neffs,'ro-')
    axes_neff[1].set_title("Neffs Iterations")
    axes_neff[1].set_xlabel("Iteration")
    fig_neff.tight_layout()
    fig_neff.savefig(os.path.join(img_savepath,"MPPI_AIS_neff_analysis.png"))

