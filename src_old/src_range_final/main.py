from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jaxopt import ScipyMinimize




import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from sklearn.covariance import OAS

from tqdm import tqdm
from time import time

import numpy as np
import jax
import jax.numpy as jnp

from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,Single_FIM_Radar,FIM_Visualization
from src_range.control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator
from src_range_final.control.MPPI import MPPI_scores_wrapper,weighting,MPPI_wrapper
from src_range_final.objective_fns.objectives import *
from src_range_final.tracking import Particle_Filter

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

    # ==================== EXPERIMENT CONFIGURATION ======================== #
    update_steps = 0

    frame_skip = 1
    tail_size = 5
    plot_size = 15
    T = 0.1
    NT = 115
    MPPI_FLAG = True
    PRUNE_FLAG = False
    MPPI_VISUALIZE = True
    MPPI_ITER_VISUALIZE = True

    N = 6
    colors = plt.cm.jet(np.linspace(0, 1, N))


    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e8;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    # alpha = (jnp.pi)**2 / 3
    B = 0.05 * 10**5

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 100

    Pt = 1000
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    Pr = K / (R ** 4)

    # get the power of the noise of the signalf
    SNR=-0


    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    time_steps = 15
    time_step_size = 0.1
    control_constraints = UNI_DI_U_LIM
    kinematic_model = unicycle_kinematics_double_integrator
    kinematic_model_vmap = vmap(kinematic_model, (0, 0, None))


    # ==================== MPPI CONFIGURATION ================================= #
    v_std = 25
    av_std = jnp.pi/180 * 45
    cov_timestep = jnp.array([[v_std**2,0],[0,av_std**2]])
    cov_traj = jax.scipy.linalg.block_diag(*[cov_timestep for t in range(time_steps)])
    cov = jax.scipy.linalg.block_diag(*[cov_traj for n in range(N)])
    # cov = jnp.stack([cov_traj for n in range(N)])
    v_init = 0
    av_init = 0 * jnp.pi/180
    mu = jnp.tile(jnp.array([v_init,av_init]),(N*time_steps,))

    num_traj = 250
    MPPI_iterations = 25

    MPPI_method = "single"
    mpc_method = "Single_FIM_3D_action_MPPI"
    u_ptb_method = "normal"
    fim_method = "Standard FIM"


    # ==================== AIS CONFIGURATION ================================= #
    temperature = 0.1
    elite_threshold = 0.9
    AIS_method = "CE"

    from copy import deepcopy
    key, subkey = jax.random.split(key)
    #
    ps = jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100)
    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    vs = jnp.zeros((ps.shape[0],1))
    avs = jnp.zeros((ps.shape[0],1))
    radar_state = jnp.column_stack((ps,chis,vs,avs))

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

    # ============================ PF CONFIGURATION ======================================#
    NP = 250
    P0_singular = jnp.diag(jnp.array([0.1, 0.1, 0.1, 1, 1, 1]));
    P0 = jnp.kron(jnp.eye(M), P0_singular)
    PF_initial_position_error = 1e-2
    PF_initial_velocity_error = 0.5
    PF_measurement_freq = 1

    MeasurementNextParallel = vmap(Particle_Filter.RangeVelocityMeasure, in_axes=(0, None))

    # ============================ MPC CONFIGURATION =====================================#
    gamma = 0.9
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

    sigmaQ = np.sqrt(10 ** 0)
    sigmaV = jnp.sqrt(1)

    A_single = jnp.array([[1., 0, 0, time_step_size, 0, 0],
                   [0, 1., 0, 0, time_step_size, 0],
                   [0, 0, 1, 0, 0, time_step_size],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(time_step_size ** 4) / 4, 0, 0, (time_step_size ** 3) / 2, 0, 0],
        [0, (time_step_size ** 4) / 4, 0, 0, (time_step_size** 3) / 2, 0],
        [0, 0, (time_step_size**4)/4, 0, 0, (time_step_size**3) / 2],
        [(time_step_size ** 3) / 2, 0, 0, (time_step_size ** 2), 0, 0],
        [0, (time_step_size ** 3) / 2, 0, 0, (time_step_size ** 2), 0],
        [0, 0, (time_step_size**3) / 2, 0, 0, (time_step_size**2)]
    ]) * sigmaQ ** 2

    A = jnp.kron(jnp.eye(M), A_single);
    Q = jnp.kron(jnp.eye(M), Q_single);
    G = jnp.eye(N)

    nx = Q.shape[0]

    J = jnp.eye(dm*M)

    Qinv = jnp.linalg.inv(Q+jnp.eye(dm*M)*1e-8)

    if fim_method == "PCRLB":
        IM_fn = partial(Single_JU_FIM_Radar, A=A, Qinv=Qinv, Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, fc=fc, c=c,
                        sigmaV=sigmaV, sigmaW=sigmaW)
    elif fim_method == "Standard FIM":
        IM_fn = partial(Single_FIM_Radar,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW)

    # IM_fn_parallel = vmap(IM_fn, in_axes=(None, 0, 0))

    MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,time_step_size=time_step_size,gamma=gamma,method=mpc_method)

    MPPI_scores = MPPI_scores_wrapper(MPC_obj,method=MPPI_method)

    MPPI_scores_vmap = jit(vmap(MPPI_scores,in_axes=(None,0,None,None,None)))

    MPPI = MPPI_wrapper(kinematic_model=kinematic_model,time_step_size=time_step_size)

    weight_fn = partial(weighting(AIS_method),elite_threshold=elite_threshold)

    WeightUpdatePF =  partial(Particle_Filter.weight_update,C=C,sigmaV=sigmaV,M=M,N=N,dm=dm,dn=dn)

    # weight_info =partial(weighting("information"),temperature=temperature)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    # time_step_sizes = jnp.tile(time_step_size, (N, 1))

    collision_penalty_vmap = jit( vmap(collision_penalty, in_axes=(0, None, None, None)))
    self_collision_penalty_vmap = jit(vmap(self_collision_penalty, in_axes=(0, None, None)))
    speed_penalty_vmap = jit(vmap(speed_penalty, in_axes=(0, None)))

    cost_collision_r2t_vmap = jit(vmap(lambda radar_states_MPPI, target_states_rollout: jnp.sum(jnp.column_stack([
                                                                                                                     collision_penalty_vmap(
                                                                                                                         radar_states_MPPI[
                                                                                                                         :,
                                                                                                                         :,
                                                                                                                         t,
                                                                                                                         :2],
                                                                                                                         target_states_rollout[
                                                                                                                             t - 1],
                                                                                                                         R_sensors_to_targets,
                                                                                                                         spread_target)
                                                                                                                     for
                                                                                                                     t
                                                                                                                     in
                                                                                                                     range(
                                                                                                                         1,
                                                                                                                         time_steps + 1)]) * (
                                                                                                            gamma ** (
                                                                                                        jnp.arange(
                                                                                                            time_steps))) / jnp.sum(
        gamma ** (jnp.arange(time_steps))), axis=-1), in_axes=(None, 0)))

    FIMs = []

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

    key, qs_simulated = Particle_Filter.generate_trajectories(key, qs, A=A, Q=Q,
                                   Gt=Gt,Gr=Gr,Pt=Pt,lam=lam,rcs=rcs,L=L,c=c,fc=fc,sigmaW=sigmaW,sigmaV=sigmaV,
                                   TN=NT)

    velocity_noises = jax.random.multivariate_normal(subkey, mean=jnp.zeros(dm // 2 * M * N, ),
                                                    cov=jnp.eye(dm // 2 * M * N) * sigmaV ** 2, shape=(NT,))


    qs_prev_noisy= qs.at[:, :dm//2].add(np.random.randn(M,dm//2)*PF_initial_position_error)
    qs_prev_noisy = qs_prev_noisy.at[:, dm//2:].add(np.random.randn(M,dm//2)*PF_initial_velocity_error)
    qs_prev_true = qs

    qs_prev_PF = jax.random.multivariate_normal(subkey, qs_prev_noisy.ravel(), P0, shape=(NP,), method="cholesky")
    weights_prev_PF = jnp.ones((NP, 1)) * 1 / NP

    for k in range(0,NT):
        mppi_round_time_start = time()
        # # get the measurement at y_(k+1)
        # measurement_next = YT[k].reshape(-1, 1)

        key, m0_particle_rollout = Particle_Filter.propogate_optimal_importance_dist_sample(key, qs_prev_PF, A, Q, time_steps)
        # sample next target state with state transition matrix x_k+1
        key, qs_next_PF = Particle_Filter.optimal_importance_dist_sample(key, qs_prev_PF, A, Q)
        qs_next_PF_copy = deepcopy(qs_next_PF)
        target_states_PF_rollout = m0_particle_rollout.reshape(NP, time_steps, M, -1)

        fig_particle,ax_particle = plt.subplots(1,1,figsize=(10,10))
        ax_particle.plot(target_states_PF_rollout[:, :, :,0].ravel(),
                 target_states_PF_rollout[:, :, :,1].ravel(), 'r.',alpha=0.01)


        target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t) @ qs_prev_true.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])


        ax_particle.plot(target_states_rollout[:, :,0], target_states_rollout[:, :, 1], 'g.')
        ax_particle.plot(m0_particle_rollout.reshape(NP, time_steps, M, -1).mean(axis=0)[:, :, 0].ravel(),
                 m0_particle_rollout.reshape(NP, time_steps, M, -1).mean(axis=0)[:, :, 1].ravel(), 'y.', alpha=0.5)

        ax_particle.plot(qs[:,0],qs[:,1],'m.')

        fig_particle.show()
        # Generate the expected Mesaurements for x_(k+1) (v_k+1)

        print(f"\n Step {k} MPPI Iteration: ")
        qs_next_true = qs_simulated[k].reshape(M,dm)
        # m0 = (A @ m0.reshape(-1, 1)).reshape(M, dm)

        best_mppi_iter_score = np.inf

        # target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])

        U_prime = deepcopy(U)
        cov_prime = deepcopy(cov)
        best_cost = jnp.inf
        for mppi_iter in range(MPPI_iterations):
            key, subkey = jax.random.split(key)

            E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime, shape=(num_traj,),method="svd")

            # simulate the model with the trajectory noise samples
            V = U_prime + E.reshape(num_traj,N,time_steps,2)

            radar_states,radar_states_MPPI = MPPI(U_nominal=U_prime,
                                                               U_MPPI=V,radar_state=radar_state)

            cost_time = time()
            # Score all the rollouts
            cost_trajectory = jnp.sum(MPPI_scores_vmap(radar_state, target_states_PF_rollout, V,A,J)*weights_prev_PF.reshape(-1,1),axis=0)

            cost_time = time() - cost_time



            cost_collision_r2t = jnp.sum(cost_collision_r2t_vmap(radar_states_MPPI,target_states_PF_rollout)*weights_prev_PF.reshape(-1,1),axis=0)#jit(vmap(lambda radar_states_MPPI,target_states_rollout: jnp.sum(jnp.column_stack([collision_penalty_vmap(radar_states_MPPI[:,:,t,:2],target_states_rollout[t-1],R_sensors_to_targets,spread_target) for t in range(1,time_steps+1)]) * (gamma**(jnp.arange(time_steps))) / jnp.sum(gamma**(jnp.arange(time_steps))) , axis=-1),in_axes=(None,0)))

            cost_collision_r2r = jnp.sum(jnp.column_stack([self_collision_penalty_vmap(radar_states_MPPI[:,:,t,:2],R_sensors_to_sensors,spread_radar) for t in range(1,time_steps+1)]) * (gamma**(jnp.arange(time_steps))) / jnp.sum(gamma**(jnp.arange(time_steps))), axis=-1)

            cost_control = ((U_prime - U).reshape(1,1,-1) @ jnp.linalg.inv(cov) @ (V).reshape(num_traj, -1,1)).ravel()

            cost_speed = jnp.sum(jnp.column_stack([speed_penalty_vmap(V[:,:,t,0],speed_minimum) for t in range(time_steps)]) * (gamma**(jnp.arange(time_steps))) / jnp.sum(gamma**(jnp.arange(time_steps))) , axis=-1)


            # print(speed_cost.mean(),speed_cost.mean()*alpha2)

            cost_MPPI = alpha1*cost_trajectory + alpha2*cost_collision_r2t + alpha3 * cost_collision_r2r * temperature * (1-alpha4) * cost_control + alpha5*cost_speed


            weights = weight_fn(cost_MPPI)


            if jnp.isnan(cost_MPPI).any():
                print("BREAK!")
                break


            if (mppi_iter < (MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):
                # U += jnp.sum(delta_actions * weights.reshape(-1, 1, 1, 1), axis=0)
                best_cost = jnp.sum(cost_MPPI*weights)
                neffs_debug[mppi_iter] = 1/ (jnp.sum(weights**2))
                U_copy = deepcopy(U_prime)
                U_prime = U_prime + jnp.sum(weights.reshape(num_traj,1,1,1) * E.reshape(num_traj,N,time_steps,2),axis=0)

                oas = OAS(assume_centered=True).fit(E[weights != 0])
                cov_prime = jnp.array(oas.covariance_)


                if k==0:
                    print(f"MPPI Shrinkage {oas.shrinkage_} @ Subiter={mppi_iter}")

            #
            if k==0 and MPPI_ITER_VISUALIZE:
                # save subiterations of MPPI
                file_mppi_debug = os.path.join(tmp_img_savepath, f"MPPI_subiteration_{mppi_iter}.png")
                images_mppi_debug.append(file_mppi_debug)
                axes_mppi_debug.plot(qs_prev_true[:,0], qs_prev_true[:,1], 'g.',label="Target Init Position")
                axes_mppi_debug.plot(qs_next_true[:,0], qs_next_true[:,1], 'go',label="Target Position")
                radar_states_debug = kinematic_model_vmap(U,np.expand_dims(radar_state, 1), time_step_size)
                radar_states_best = kinematic_model_vmap(U_prime, jnp.expand_dims(radar_state, 1),time_step_size)

                if MPPI_VISUALIZE:
                    # for n in range(N):
                    mppi_colors = (cost_MPPI - cost_MPPI.min())/(cost_MPPI.ptp())
                    mppi_color_idx = np.argsort(mppi_colors)[::-1]
                    segs = radar_states_MPPI[mppi_color_idx].reshape(num_traj*N,time_steps+1,-1,order='F')
                    segs = LineCollection(segs[:,:,:2],colors=plt.cm.jet(np.tile(mppi_colors[mppi_color_idx],(N,1)).T.reshape(-1,order='F')),alpha=0.5)
                    # axes_mppi_debug.plot(radar_states_MPPI[:, n, :, 0].T, radar_states_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
                    axes_mppi_debug.add_collection(segs)

                axes_mppi_debug.plot(radar_states_debug[:,0,0], radar_states_debug[:,0,1], 'r*',label="Sensor Position")
                axes_mppi_debug.plot(radar_states_debug[:,1:,0].T, radar_states_debug[:,1:,1].T, 'r-',label="_nolegend_")
                axes_mppi_debug.plot(radar_states_best[:,1:,0].T, radar_states_best[:,1:,1].T, color="magenta",linestyle='--',label="_nolegend_")

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


        if jnp.isnan(cost_MPPI).any():
            print("BREAK!")
            break

        weights = weight_fn(cost_MPPI)

        mean_shift = (U_prime - U)

        E_prime = E + mean_shift.ravel()

        U += jnp.sum(weights.reshape(-1,1,1,1) * E_prime.reshape(num_traj,N,time_steps,2),axis=0)

        U = jnp.stack((jnp.clip(U[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)

        print("MPPI Mean Score: ",jnp.nanmean(cost_MPPI))
        print("MPPI Best Score: ",-best_mppi_iter_score)
        print(f"Vmax = {radar_state[:,2].min()} , Vmin = {radar_state[:,2].max()}")

        radar_states = kinematic_model_vmap( U ,jnp.expand_dims(radar_state, 1), time_step_size)


        # U += jnp.clip(jnp.sum(weights.reshape(num_traj,1,1,1) *  E.reshape(num_traj,N,time_steps,2),axis=0),U_lower,U_upper)
        U = jnp.roll(U,-1,axis=1)


        radar_state = radar_states[:,1]

        # get the  expected measurement at next radar statae
        target_measurement_expected = MeasurementNextParallel(target_states_PF_rollout[:,0,:,:],radar_state[:,:2])

        # get measurement at next radar state
        target_measurement_actual = Particle_Filter.RangeVelocityMeasure(qs_next_true, radar_state[:,:2])

        key, subkey = jax.random.split(key)
        # noise for the velocities
        velocity_noise = velocity_noises[k,:].reshape(M * N, dm // 2)

        range_measures = target_measurement_actual[:, 0].ravel()

        range_noise = jax.random.multivariate_normal(subkey, mean=jnp.zeros(M * N, ), cov=C * jnp.diag(range_measures ** 4),
                                            shape=()).reshape(M * N, 1)


        target_measurement_noisy = target_measurement_actual + jnp.concatenate((range_noise, velocity_noise), axis=-1)

        weights_next_PF = WeightUpdatePF(weights_prev_PF, target_measurement_expected,target_measurement_noisy)

        neffs_PF = Particle_Filter.effective_samples(weights_next_PF)

        # if neffs_PF < (NP * 2 / 3) or ((k + 1) % 250) == 0:
            # print("\n Particle Resampling")
        qs_next_PF, weights_next_PF = Particle_Filter.weight_resample(qs_next_PF, weights_next_PF)

        # Filter[t] = (Wnext * Xnext).sum(axis=0)

        J = IM_fn(radar_state=radar_state,target_state=qs_next_true,J=J) #[JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],s=s) for i in range(len(Js))]

         # print(jnp.trace(J))
        FIMs.append(jnp.linalg.slogdet(J)[1].ravel())
        file = os.path.join(tmp_img_savepath,f"JU_test_target_movement{k}.png")
        cost_means[k] = cost_MPPI.mean()
        cost_stds[k] = cost_MPPI.std()
        neffs[k] = 1 / (jnp.sum(weights ** 2))

        if ((k+1) % frame_skip) == 0:
            fig_time = time()
            # for n in range(N):
            #     axes[0].plot(P_MPPI[:,n,:,0].T,P_MPPI[:,n,:,1].T,'b-',label="__nolegend__")

            # Main figure
            file_main = os.path.join(tmp_img_savepath, f"JU_test_target_movement{k}.png")
            images_main.append(file_main)
            axes_main[0].plot(qs_prev_true[:,0], qs_prev_true[:,1], 'g.',label="Target Init Position")
            axes_main[0].plot(qs_next_true[:,0], qs_next_true[:,1], 'go',label="Target Position")
            axes_main[0].plot(ps_init[:,0], ps_init[:,1], 'md',label="Sensor Init")
            axes_main[0].plot(qs_next_PF_copy.reshape(-1, M, dm)[:, :, 0].ravel(), qs_next_PF_copy.reshape(-1, M, dm)[:, :, 1].ravel(), 'g.', label="PF Prop", alpha=0.1)

            for m in range(M):
                axes_main[0].add_patch(
                    Circle(qs_next_true[m, :], radius_projected, edgecolor="green", fill=False, lw=1,
                           linestyle="--", label="_nolegend_"))

            for n in range(N):
                axes_main[0].add_patch(
                    Circle(radar_states[n,1, :2], R_sensors_to_sensors, edgecolor="red", fill=False, lw=1,
                           linestyle="--", label="_nolegend_"))

            # if MPPI_VISUALIZE:
            #     for n in range(N):
            #         axes_main[0].plot(radar_states_MPPI[:, n, :, 0].T, radar_states_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
            if MPPI_VISUALIZE:
                # for n in range(N):
                mppi_colors = (cost_MPPI - cost_MPPI.min()) / (cost_MPPI.ptp())
                mppi_color_idx = np.argsort(mppi_colors)[::-1]
                segs = radar_states_MPPI[mppi_color_idx].reshape(num_traj * N, time_steps + 1, -1, order='F')
                segs = LineCollection(segs[:, :, :2], colors=plt.cm.jet(
                    np.tile(mppi_colors[mppi_color_idx], (N, 1)).T.reshape(-1, order='F')), alpha=0.5)
                # axes_mppi_debug.plot(radar_states_MPPI[:, n, :, 0].T, radar_states_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
                axes_main[0].add_collection(segs)

            axes_main[0].plot(radar_states[:,0,0], radar_states[:,0,1], 'r*',label="Sensor Position")
            axes_main[0].plot(radar_states.squeeze()[:,1:,0].T, radar_states.squeeze()[:,1:,1].T, 'r-',label="_nolegend_")
            axes_main[0].plot([],[],"r.-",label="Sensor Planned Path")
            axes_main[0].set_title(f"k={k}")
            axes_main[0].set_title(f"k={k}")
            # axes[0].legend(bbox_to_anchor=(0.5, 1.45),loc="upper center")
            axes_main[0].legend(bbox_to_anchor=(0.7, 1.45),loc="upper center")
            axes_main[0].grid()
            axes_main[0].axis('equal')

            qx,qy,logdet_grid = FIM_Visualization(ps=radar_state[:,:2], qs=qs_next_true,
                                                  Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaW=sigmaW,
                                                  N=1000)

            axes_main[1].contourf(qx, qy, logdet_grid, levels=20)
            axes_main[1].scatter(radar_state[:, 0], radar_state[:, 1], s=50, marker="x", color="r")
            #
            axes_main[1].scatter(qs_next_true[:, 0], qs_next_true[:, 1], s=50, marker="o", color="g")
            axes_main[1].set_title("Instant Time Objective Function Map")
            axes_main[1].grid()
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

        qs_prev_true = qs_next_true
        qs_prev_PF = qs_next_PF;
        weights_prev_PF = weights_next_PF

        mppi_round_time_end = time()

        print("MPPI Round Time: ",mppi_round_time_end-mppi_round_time_start)
        print("PF Neff: ",neffs_PF)
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
