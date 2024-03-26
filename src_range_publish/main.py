from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jaxopt import ScipyMinimize



from src_range_publish.FIM_new.FIM_RADAR import Single_FIM_Radar,FIM_Visualization
from src_range_publish.control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator
from src_range_publish.utils import visualize_tracking,visualize_control,visualize_target_mse
import matplotlib
matplotlib.use('Agg')
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

from src_range_publish.control.MPPI import MPPI_scores_wrapper,weighting,MPPI_wrapper #,MPPI_adapt_distribution
from src_range_publish.objective_fns.objectives import *
import src_range_publish.tracking.cubatureKalmanFilter as cubatureKalmanFilter
from src_range_publish.tracking.cubatureTestMLP import measurement_model,transition_model,generate_data_state,generate_measurement_noisy
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

    gif_savepath = os.path.join("..", "images", "gifs","main")
    tmp_img_savepath = os.path.join("tmp_images")
    img_savepath = os.path.join("..","..","images")
    os.makedirs(tmp_img_savepath,exist_ok=True)
    os.makedirs(gif_savepath,exist_ok=True)

    # =========================== Experiment Choice ================== #
    update_steps = 0

    frame_skip = 4
    tail_size = 5
    plot_size = 15
    T = 0.1
    N_steps = 1000
    MPPI_FLAG = True
    PRUNE_FLAG = False
    MPPI_VISUALIZE = True
    MPPI_ITER_VISUALIZE = True

    N_radar = 8
    colors = plt.cm.jet(np.linspace(0, 1,N_radar))

    update_freq_control = 4
    update_freq_ckf = 1

    dt_ckf = 0.025
    dt_control = 0.1

    # ==================== RADAR CONFIGURATION ======================== #
    c = 299792458
    fc = 1e8;
    Gt = 200;
    Gr = 200;
    lam = c / fc
    rcs = 1;
    L = 1;

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 500

    Pt = 1000
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    Pr = K / (R ** 4)

    # get the power of the noise of the signalf
    SNR=-20


    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    horizon = 15
    control_constraints = UNI_DI_U_LIM
    kinematic_model = unicycle_kinematics_double_integrator

    # ==================== MPPI CONFIGURATION ================================= #
    u1_std = 25
    u2_std = jnp.pi/180 * 45
    cov_timestep = jnp.array([[u1_std**2,0],[0,u2_std**2]])
    cov_traj = jax.scipy.linalg.block_diag(*[cov_timestep for _ in range(horizon)])
    cov = jax.scipy.linalg.block_diag(*[cov_traj for _ in range(N_radar)])
    # cov = jnp.stack([cov_traj for n in range(N)])
    u1_init = 0
    u2_init = 0 * jnp.pi/180

    num_traj = 250
    MPPI_iterations = 25
    MPPI_method = "single"
    mpc_method = "Single_FIM_3D_action_MPPI"
    u_ptb_method = "normal"
    fim_method = "Standard FIM"
    move_radar=True


    # ==================== AIS CONFIGURATION ================================= #
    temperature = 0.1
    elite_threshold = 0.9
    AIS_method = "CE"

    from copy import deepcopy
    key, subkey = jax.random.split(key)
    #
    ps = jnp.concatenate((jax.random.uniform(key, shape=(N_radar, 2), minval=-400, maxval=400),jnp.zeros((N_radar,1))),axis=-1)
    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    vs = jnp.zeros((ps.shape[0],1))
    avs = jnp.zeros((ps.shape[0],1))
    radar_state = jnp.column_stack((ps,chis,vs,avs))
    radar_state_init = deepcopy(radar_state)

    ps_init = deepcopy(ps)
    z_elevation = 100
    # qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
    #                 [-50.4,30.32,z_elevation,-20,-10,0], #,
    #                 # [10,10,z_elevation,10,10,0],
    #                 [20,20,z_elevation,5,-5,0]])
    # qs = jnp.array([[0.0, -0.0,z_elevation, 0., 0,0], #,#,
    #                 [-50.4,30.32,z_elevation,-0,-0,0], #,
    #                 [10,10,z_elevation,0,0,0],
    #                 [20,20,z_elevation,0,0,0]])
    target_state = jnp.array([[0.0, -0.0,z_elevation+10, 25., 20,0], #,#,
                    [-100.4,-30.32,z_elevation-15,20,-10,0], #,
                    [30,30,z_elevation+20,-10,-10,0]])#,

    M_target, dm = target_state.shape;
    _ , dn = radar_state.shape;

    sigmaW = jnp.sqrt(M_target*Pr/ (10**(SNR/10)))
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
    speed_minimum = 5
    R_sensors_to_targets = 125
    R_sensors_to_sensors = 10

    alpha1 = 1 # FIM
    alpha2 = 80 # Target - Radar Distance
    alpha3 = 60 # Radar - Radar Distance
    alpha4 = 1 # AIS control cost
    alpha5 = 0 # speed cost

    thetas = jnp.arcsin(target_state[:,2]/R_sensors_to_targets)
    radius_projected = R_sensors_to_targets * jnp.cos(thetas)

    # ========================= Target State Space ============================== #

    sigmaQ = np.sqrt(10 ** 1)

    A_single = jnp.array([[1., 0, 0, dt_control, 0, 0],
                   [0, 1., 0, 0, dt_control, 0],
                   [0, 0, 1, 0, 0, dt_control],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(dt_control ** 4) / 4, 0, 0, (dt_control ** 3) / 2, 0, 0],
        [0, (dt_control ** 4) / 4, 0, 0, (dt_control** 3) / 2, 0],
        [0, 0, (dt_control**4)/4, 0, 0, (dt_control**3) / 2],
        [(dt_control ** 3) / 2, 0, 0, (dt_control ** 2), 0, 0],
        [0, (dt_control ** 3) / 2, 0, 0, (dt_control ** 2), 0],
        [0, 0, (dt_control**3) / 2, 0, 0, (dt_control**2)]
    ]) * sigmaQ ** 2

    A = jnp.kron(jnp.eye(M_target), A_single);
    Q = jnp.kron(jnp.eye(M_target), Q_single);

    nx = Q.shape[0]

    # ============================ CKF Settings =====================================#
    ckf = cubatureKalmanFilter.CubatureKalmanFilter(dim_x=M_target*dm, dim_z=M_target*N_radar, dt=dt_ckf,
                                                     hx=measurement_model, fx=transition_model)

    # target_state_est_ckf = np.zeros((dm*M_target, NT))
    ckf.x = np.array(target_state).ravel() + np.random.randn(M_target*dm) * 10
    ckf.x = ckf.x.reshape(-1,1)
    ckf.P = np.eye(M_target*dm) * 50

    Q_single_ckf = jnp.array([
        [(dt_ckf ** 4) / 4, 0, 0, (dt_ckf ** 3) / 2, 0, 0],
        [0, (dt_ckf ** 4) / 4, 0, 0, (dt_ckf** 3) / 2, 0],
        [0, 0, (dt_ckf**4)/4, 0, 0, (dt_ckf**3) / 2],
        [(dt_ckf ** 3) / 2, 0, 0, (dt_ckf ** 2), 0, 0],
        [0, (dt_ckf ** 3) / 2, 0, 0, (dt_ckf ** 2), 0],
        [0, 0, (dt_ckf**3) / 2, 0, 0, (dt_ckf**2)]
    ]) * sigmaQ ** 2

    Q_ckf = jnp.kron(jnp.eye(M_target), Q_single_ckf);

    ckf.Q = Q_ckf

    # ======================== Objective Settings ===============================#

    J = jnp.eye(dm*M_target) #jnp.stack([jnp.eye(d) for m in range(M)])

    Qinv = jnp.linalg.solve(Q,jnp.eye(Q.shape[0])) #+jnp.eye(dm*M)*1e-8)

    IM_fn = partial(Single_FIM_Radar,C=C)


    MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=dt_control,gamma=gamma,method=mpc_method)

    MPPI_scores = MPPI_scores_wrapper(MPC_obj,method=MPPI_method)

    MPPI = MPPI_wrapper(kinematic_model=kinematic_model,dt=dt_control)

    if AIS_method == "CE":
        weight_fn = partial(weighting(AIS_method),elite_threshold=elite_threshold)
    elif AIS_method == "information":
        weight_fn = partial(weighting(AIS_method),temperature=temperature)

    # weight_info =partial(weighting("information"),temperature=temperature)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    # dt_controls = jnp.tile(dt_control, (N, 1))

    collision_penalty_vmap = jit( vmap(collision_penalty, in_axes=(0, None, None)))
    self_collision_penalty_vmap = jit(vmap(self_collision_penalty, in_axes=(0, None)))
    speed_penalty_vmap = jit(vmap(speed_penalty, in_axes=(0, None)))


    U1 = jnp.ones((N_radar,horizon,1)) * u1_init
    U2 = jnp.ones((N_radar,horizon,1)) * u2_init
    U =jnp.concatenate((U1,U2),axis=-1)

    if not move_radar:
        U = jnp.zeros_like(U)
        radar_states_MPPI = None
        cost_MPPI = None
    # generate radar states at measurement frequency
    radar_states = kinematic_model(np.repeat(U, update_freq_control, axis=1)[:, :update_freq_control :],
                                   radar_state, dt_ckf)

    # U += jnp.clip(jnp.sum(weights.reshape(num_traj,1,1,1) *  E.reshape(num_traj,N,horizon,2),axis=0),U_lower,U_upper)

    # generate the true target state
    target_states_true = jnp.array(generate_data_state(target_state,N_steps, M_target, dm,dt=dt_ckf))


    FIMs = np.zeros(N_steps//update_freq_control + 1)


    fig_main,axes_main = plt.subplots(1,3,figsize=(15,5))
    imgs_main =  []

    fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))
    imgs_control =  []


    fig_mse,axes_mse = plt.subplots(1,figsize=(10,5))
    target_state_mse = np.zeros(N_steps)

    for step in range(1,N_steps):
        target_state_true = target_states_true[:, step-1].reshape(M_target,dm)

        if (step % update_freq_ckf) == 0:
            print(f"Step {step} - CKF Update")
            ckf.predict(fx_args=(M_target,))
            target_state_pred = jnp.array(ckf.x).reshape(M_target,dm)


        best_mppi_iter_score = np.inf
        mppi_round_time_start = time()

        # need dimension Horizon x Number of Targets x Dim of Targets
        # target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M_target * dm).T).T.reshape(M_target, dm) for t in range(1,horizon+1)])

        if move_radar:
            if (step % update_freq_control == 0):
                # the cubature kalman filter points propogated over horizon. Horizon x # Sigma Points (2*dm) x (Number of targets * dim of target)
                target_states_ckf = ckf.predict_propogate(ckf.x, ckf.P, horizon, dt=dt_control, fx_args=(M_target,))
                target_states_ckf = np.swapaxes(target_states_ckf.mean(axis=1).reshape(horizon, M_target, dm), 1, 0)


                mppi_start_time = time()

                U_prime = deepcopy(U)
                cov_prime = deepcopy(cov)

                print(f"\n Step {step} MPPI CONTROL ")


                for mppi_iter in range(MPPI_iterations):
                    start = time()
                    key, subkey = jax.random.split(key)

                    mppi_start = time()

                    E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime, shape=(num_traj,),method="svd")

                    # simulate the model with the trajectory noise samples
                    V = U_prime + E.reshape(num_traj,N_radar,horizon,2)

                    mppi_rollout_start = time()

                    radar_states,radar_states_MPPI = MPPI(U_nominal=U_prime,
                                                                       U_MPPI=V,radar_state=radar_state)

                    mppi_rollout_end = time()


                    # GET MPC OBJECTIVE
                    mppi_score_start = time()
                    # Score all the rollouts
                    cost_trajectory = MPPI_scores(radar_state, target_states_ckf, V,
                                              A=A,J=J)

                    mppi_score_end = time()


                    cost_collision_r2t = collision_penalty_vmap(radar_states_MPPI[...,1:horizon+1,:], target_states_ckf,
                                           R_sensors_to_targets)

                    cost_collision_r2t = jnp.sum((cost_collision_r2t * gamma**(jnp.arange(horizon))) / jnp.sum(gamma**jnp.arange(horizon)),axis=-1)

                    cost_collision_r2r = self_collision_penalty_vmap(radar_states_MPPI[...,1:horizon+1,:], R_sensors_to_sensors)
                    cost_collision_r2r = jnp.sum((cost_collision_r2r * gamma**(jnp.arange(horizon))) / jnp.sum(gamma**jnp.arange(horizon)),axis=-1)

                    cost_speed = speed_penalty_vmap(V[...,0], speed_minimum)
                    cost_speed = jnp.sum((cost_speed * gamma**(jnp.arange(horizon))) / jnp.sum(gamma**jnp.arange(horizon)),axis=-1)


                    cost_control = ((U_prime - U).reshape(1,1,-1) @ jnp.linalg.inv(cov) @ (V).reshape(num_traj, -1,1)).ravel()

                    cost_MPPI = alpha1*cost_trajectory + alpha2*cost_collision_r2t + alpha3 * cost_collision_r2r * temperature * (1-alpha4) * cost_control + alpha5*cost_speed


                    weights = weight_fn(cost_MPPI)


                    if jnp.isnan(cost_MPPI).any():
                        print("BREAK!")
                        break

                    if (mppi_iter < (MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):

                        best_cost = jnp.sum(cost_MPPI*weights)

                        U_copy = deepcopy(U_prime)
                        U_prime = U_prime + jnp.sum(weights.reshape(num_traj,1,1,1) * E.reshape(num_traj,N_radar,horizon,2),axis=0)

                        oas = OAS(assume_centered=True).fit(E[weights != 0])
                        cov_prime = jnp.array(oas.covariance_)


                mppi_round_time_end = time()

                if jnp.isnan(cost_MPPI).any():
                    print("BREAK!")
                    break

                weights = weight_fn(cost_MPPI)

                mean_shift = (U_prime - U)

                E_prime = E + mean_shift.ravel()

                U += jnp.sum(weights.reshape(-1,1,1,1) * E_prime.reshape(num_traj,N_radar,horizon,2),axis=0)

                U = jnp.stack((jnp.clip(U[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)

                # jnp.repeat(U,update_freq_control,axis=1)

                # radar_states = kinematic_model(U ,radar_state, dt_control)

                # generate radar states at measurement frequency
                radar_states = kinematic_model(np.repeat(U, update_freq_control, axis=1)[:, :update_freq_control, :],
                                               radar_state, dt_ckf)

                # U += jnp.clip(jnp.sum(weights.reshape(num_traj,1,1,1) *  E.reshape(num_traj,N,horizon,2),axis=0),U_lower,U_upper)

                # radar_state = radar_states[:,1]
                U = jnp.roll(U, -1, axis=1)

                mppi_end_time = time()
                print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))



        if step >= update_freq_control:
            # get the radar state
            # print("Here",step,(step%update_freq_control)+1)
            radar_state = radar_states[:,(step%update_freq_control)+1,:]

        J = IM_fn(radar_state=radar_state, target_state=ckf.x.reshape(M_target,dm),
                  J=J)

        FIMs[step // update_freq_control - 1] = jnp.linalg.slogdet(J)[1].ravel()

        if (step % frame_skip) == 0 and (step % update_freq_control) == 0:
            print(f"Step {step} - Saving Figure ")

            axes_main[0].plot(radar_state_init[:, 0], radar_state_init[:, 1], 'mo',
                     label="Radar Init")

            imgs_main.append(visualize_tracking(target_state_true=target_state_true, target_state_ckf=ckf.x.reshape(M_target,dm),
                           radar_state=radar_state,radar_states_MPPI=radar_states_MPPI,
                           cost_MPPI=cost_MPPI, FIMs=FIMs[:(step//update_freq_control)],
                           R2T=radius_projected, R2R=R_sensors_to_sensors,C=C,
                           fig=fig_main, axes=axes_main, step=step,
                           tmp_photo_dir = tmp_img_savepath, filename = "MPPI_CKF"))


            imgs_control.append(visualize_control(U=jnp.roll(U,1,axis=1),CONTROL_LIM=control_constraints,
                           fig=fig_control, axes=axes_control, step=step,
                           tmp_photo_dir = tmp_img_savepath, filename = "MPPI_control"))



        # J = IM_fn(radar_state=radar_state,target_state=m0,J=J)

        # CKF ! ! ! !
        measurement_next_expected = measurement_model(ckf.x.ravel(), radar_state[:,:3], M_target, dm,N_radar)

        R = np.diag(C * (measurement_next_expected/2) ** 4)
        ckf.R = R

        range_actual = measurement_model(target_states_true[:, step-1], radar_state[:, :3], M_target, dm, N_radar)

        measurement_actual = range_actual + np.random.randn() * (C * (range_actual.ravel() / 2) ** 4)

        ckf.update(np.reshape(measurement_actual,(-1,1)), hx_args=(radar_state[:,:3], M_target, dm,N_radar))
        print(f"Step {step} - Tracking ")

        target_state_mse[step-1] = jnp.sqrt(jnp.sum(ckf.x - target_state_true.reshape(-1,1))**2)


    visualize_target_mse(target_state_mse,fig_mse,axes_mse,gif_savepath,filename="target_mse")

    images = [imageio.imread(file) for file in imgs_main]
    imageio.mimsave(os.path.join(gif_savepath, f'MPPI_MPC_AIS={AIS_method}_FIM={fim_method}.gif'), images, duration=0.1)

    images = [imageio.imread(file) for file in imgs_control]
    imageio.mimsave(os.path.join(gif_savepath, f'MPPI_Control_AIS={AIS_method}.gif'), images, duration=0.1)
