from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import jax
import jax.numpy as jnp

from sklearn.covariance import OAS

from FIM_new.FIM_RADAR import Single_JU_FIM_Radar,JU_RANGE_SFIM,Single_FIM_Radar,FIM_Visualization
from control.Sensor_Dynamics import UNI_SI_U_LIM,UNI_DI_U_LIM,unicycle_kinematics_single_integrator,unicycle_kinematics_double_integrator
from utils import visualize_tracking,visualize_control,visualize_target_mse,place_sensors_restricted
from control.MPPI import MPPI_scores_wrapper,weighting,MPPI_wrapper #,MPPI_adapt_distribution
from objective_fns.objectives import *
import tracking.cubatureKalmanFilter as cubatureKalmanFilter
from tracking.cubatureTestMLP import measurement_model,transition_model,generate_data_state,generate_measurement_noisy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as mplstyle

import imageio

from time import time
import os
import argparse
from copy import deepcopy
import shutil

def main(args):
    mpl.rcParams['path.simplify_threshold'] = 1.0
    mplstyle.use('fast')
    mplstyle.use(['ggplot', 'fast'])
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # =========================== Experiment Choice ================== #
    update_freq_control = int(args.dt_control/args.dt_ckf) #4
    update_freq_ckf = 1

    # ==================== RADAR setup ======================== #
    # speed of light
    c = 299792458
    K = args.Pt * args.Gt * args.Gr * (c/args.fc) ** 2 * args.rcs / args.L / (4 * jnp.pi) ** 3
    Pr = K / (args.R ** 4)

    # ==================== SENSOR DYNAMICS CONFIGURATION ======================== #
    control_constraints = UNI_DI_U_LIM
    kinematic_model = unicycle_kinematics_double_integrator

    # ==================== MPPI CONFIGURATION ================================= #
    cov_timestep = jnp.array([[args.acc_std**2,0],[0,args.ang_acc_std**2]])
    cov_traj = jax.scipy.linalg.block_diag(*[cov_timestep for _ in range(args.horizon)])
    cov = jax.scipy.linalg.block_diag(*[cov_traj for _ in range(args.N_radar)])
    # cov = jnp.stack([cov_traj for n in range(N)])

    mpc_method = "Single_FIM_3D_action_MPPI"

    # ==================== AIS CONFIGURATION ================================= #
    key, subkey = jax.random.split(key)
    #

    z_elevation = 60
    # target_state = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
    #                 [-50.4,30.32,z_elevation,-20,-10,0], #,
    #                 # [10,10,z_elevation,10,10,0],
    #                 [20,20,z_elevation,5,-5,0]])
    target_state = jnp.array([[0.0, -0.0,z_elevation-5, 20., 10,0], #,#,
                    [15.4,15.32,z_elevation+10,15,20,0], #,
                    [10,10,z_elevation-5,17,19,0],
                    [20,20,z_elevation-15,6,8,0]])
    # target_state = jnp.array([[0.0, -0.0,z_elevation+10, 25., 20,0], #,#,
    #                 [-100.4,-30.32,z_elevation-15,20,-10,0], #,
    #                 [30,30,z_elevation+20,-10,-10,0]])#,

    ps,key = place_sensors_restricted(key,target_state,args.R2R,args.R2T,-400,400,args.N_radar)
    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi)
    vs = jnp.zeros((ps.shape[0],1))
    avs = jnp.zeros((ps.shape[0],1))
    radar_state = jnp.column_stack((ps,chis,vs,avs))
    radar_state_init = deepcopy(radar_state)


    M_target, dm = target_state.shape;
    _ , dn = radar_state.shape;

    sigmaW = jnp.sqrt(M_target*Pr/ (10**(args.SNR/10)))
    # coef = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi)** 3 / (R ** 4)
    C = c**2 * sigmaW**2 / (jnp.pi**2 * 8 * args.fc**2) * 1/K

    print("Noise Power: ",sigmaW**2)
    print("Power Return (RCS): ",Pr)
    print("K",K)

    print("Pt (peak power)={:.9f}".format(args.Pt))
    print("lam ={:.9f}".format(c/args.fc))
    print("C=",C)

    # ========================= Target State Space ============================== #

    sigmaQ = np.sqrt(10 ** 1)

    A_single = jnp.array([[1., 0, 0, args.dt_control, 0, 0],
                   [0, 1., 0, 0, args.dt_control, 0],
                   [0, 0, 1, 0, 0, args.dt_control],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(args.dt_control ** 4) / 4, 0, 0, (args.dt_control ** 3) / 2, 0, 0],
        [0, (args.dt_control ** 4) / 4, 0, 0, (args.dt_control** 3) / 2, 0],
        [0, 0, (args.dt_control**4)/4, 0, 0, (args.dt_control**3) / 2],
        [(args.dt_control ** 3) / 2, 0, 0, (args.dt_control ** 2), 0, 0],
        [0, (args.dt_control ** 3) / 2, 0, 0, (args.dt_control ** 2), 0],
        [0, 0, (args.dt_control**3) / 2, 0, 0, (args.dt_control**2)]
    ]) * sigmaQ ** 2

    A = jnp.kron(jnp.eye(M_target), A_single);
    Q = jnp.kron(jnp.eye(M_target), Q_single);
    # Q = Q + jnp.eye(Q.shape[0])*1e-6
    #
    nx = Q.shape[0]

    # ============================ CKF Settings =====================================#
    ckf = cubatureKalmanFilter.CubatureKalmanFilter(dim_x=M_target*dm, dim_z=M_target*args.N_radar, dt=args.dt_ckf,
                                                     hx=measurement_model, fx=transition_model)

    # target_state_est_ckf = np.zeros((dm*M_target, NT))
    ckf.x = np.array(target_state).ravel() + np.random.randn(M_target*dm) * 10
    ckf.x = ckf.x.reshape(-1,1)
    ckf.P = np.eye(M_target*dm) * 50

    A_single_ckf = jnp.array([[1., 0, 0, args.dt_ckf, 0, 0],
                   [0, 1., 0, 0, args.dt_ckf, 0],
                   [0, 0, 1, 0, 0, args.dt_ckf],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1., 0],
                   [0, 0, 0, 0, 0, 1]])

    Q_single_ckf = jnp.array([
        [(args.dt_ckf ** 4) / 4, 0, 0, (args.dt_ckf ** 3) / 2, 0, 0],
        [0, (args.dt_ckf ** 4) / 4, 0, 0, (args.dt_ckf** 3) / 2, 0],
        [0, 0, (args.dt_ckf**4)/4, 0, 0, (args.dt_ckf**3) / 2],
        [(args.dt_ckf ** 3) / 2, 0, 0, (args.dt_ckf ** 2), 0, 0],
        [0, (args.dt_ckf ** 3) / 2, 0, 0, (args.dt_ckf ** 2), 0],
        [0, 0, (args.dt_ckf**3) / 2, 0, 0, (args.dt_ckf**2)]
    ]) * sigmaQ ** 2

    A_ckf = jnp.kron(jnp.eye(M_target), A_single_ckf);
    Q_ckf = jnp.kron(jnp.eye(M_target), Q_single_ckf);

    ckf.Q = Q_ckf

    # ======================== Objective Settings ===============================#

    J = jnp.linalg.inv(ckf.P)

    if args.fim_method == "PFIM":
        IM_fn = partial(Single_JU_FIM_Radar, A=A, Q=Q, C=C)
        IM_fn_update = partial(Single_JU_FIM_Radar, A=A_ckf, Q=Q_ckf, C=C)
    elif args.fim_method == "SFIM":
        IM_fn = partial(Single_FIM_Radar, C=C)
        IM_fn_update = IM_fn
    elif args.fim_method == "SFIM_bad":
        sigmaR = 1
        IM_fn = partial(JU_RANGE_SFIM, R=jnp.eye(M_target*args.N_radar)*sigmaR**2)
        IM_fn_update = IM_fn


    MPC_obj = MPC_decorator(IM_fn=IM_fn,kinematic_model=kinematic_model,dt=args.dt_control,gamma=args.gamma,method=mpc_method)

    MPPI_scores = MPPI_scores_wrapper(MPC_obj)

    MPPI = MPPI_wrapper(kinematic_model=kinematic_model,dt=args.dt_control)

    if args.AIS_method == "CE":
        weight_fn = partial(weighting(args.AIS_method),elite_threshold=args.elite_threshold)
    elif args.AIS_method == "information":
        weight_fn = partial(weighting(args.AIS_method),temperature=args.temperature)

    # weight_info =partial(weighting("information"),temperature=temperature)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    # dt_controls = jnp.tile(dt_control, (N, 1))

    collision_penalty_vmap = jit( vmap(collision_penalty, in_axes=(0, None, None)))
    self_collision_penalty_vmap = jit(vmap(self_collision_penalty, in_axes=(0, None)))
    speed_penalty_vmap = jit(vmap(speed_penalty, in_axes=(0, None)))


    U1 = jnp.ones((args.N_radar,args.horizon,1)) #* args.acc_init
    U2 = jnp.ones((args.N_radar,args.horizon,1)) #* args.ang_acc_init
    U =jnp.concatenate((U1,U2),axis=-1)

    if not args.move_radars:
        U = jnp.zeros_like(U)
        radar_states_MPPI = None
        cost_MPPI = None

    # # generate radar states at measurement frequency
    # radar_states = kinematic_model(np.repeat(U, update_freq_control+1, axis=1)[:, :update_freq_control,:],
    #                                radar_state, args.dt_ckf)

    # U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,args.horizon,2),axis=0),U_lower,U_upper)

    # generate the true target state
    target_states_true = jnp.array(generate_data_state(target_state,args.N_steps, M_target, dm,dt=args.dt_ckf,Q=Q))

    # radar state history
    radar_state_history = np.zeros((args.N_steps//update_freq_control,)+radar_state.shape)

    FIMs = np.zeros(args.N_steps//update_freq_control + 1)


    fig_main,axes_main = plt.subplots(1,2,figsize=(10,5))
    imgs_main =  []

    fig_control,axes_control = plt.subplots(1,2,figsize=(10,5))
    imgs_control =  []


    fig_mse,axes_mse = plt.subplots(1,figsize=(10,5))
    target_state_mse = np.zeros(args.N_steps)

    for step in range(1,args.N_steps+1):
        target_state_true = target_states_true[:, step-1].reshape(M_target,dm)

        if (step % update_freq_ckf) == 0:
            print(f"Step {step} - CKF Update")
            ckf.predict(fx_args=(M_target,))
            target_state_pred = jnp.array(ckf.x).reshape(M_target,dm)


        best_mppi_iter_score = np.inf
        mppi_round_time_start = time()

        # need dimension Horizon x Number of Targets x Dim of Targets
        # target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M_target * dm).T).T.reshape(M_target, dm) for t in range(1,horizon+1)])

        if args.move_radars:
            if (step % update_freq_control == 0):
                # the cubature kalman filter points propogated over horizon. Horizon x # Sigma Points (2*dm) x (Number of targets * dim of target)
                target_states_ckf = ckf.predict_propogate(ckf.x, ckf.P, args.horizon, dt=args.dt_control, fx_args=(M_target,))
                target_states_ckf = np.swapaxes(target_states_ckf.mean(axis=1).reshape(args.horizon, M_target, dm), 1, 0)


                mppi_start_time = time()

                U_prime = deepcopy(U)
                cov_prime = deepcopy(cov)

                print(f"\n Step {step} MPPI CONTROL ")


                for mppi_iter in range(args.MPPI_iterations):
                    start = time()
                    key, subkey = jax.random.split(key)

                    mppi_start = time()

                    E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime, shape=(args.num_traj,),method="svd")

                    # simulate the model with the trajectory noise samples
                    V = U_prime + E.reshape(args.num_traj,args.N_radar,args.horizon,2)

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


                    cost_collision_r2t = collision_penalty_vmap(radar_states_MPPI[...,1:args.horizon+1,:], target_states_ckf,
                                           args.R2T)

                    cost_collision_r2t = jnp.sum((cost_collision_r2t * args.gamma**(jnp.arange(args.horizon))) / jnp.sum(args.gamma**jnp.arange(args.horizon)),axis=-1)

                    cost_collision_r2r = self_collision_penalty_vmap(radar_states_MPPI[...,1:args.horizon+1,:], args.R2R)
                    cost_collision_r2r = jnp.sum((cost_collision_r2r * args.gamma**(jnp.arange(args.horizon))) / jnp.sum(args.gamma**jnp.arange(args.horizon)),axis=-1)

                    cost_speed = speed_penalty_vmap(V[...,0], args.speed_minimum)
                    cost_speed = jnp.sum((cost_speed * args.gamma**(jnp.arange(args.horizon))) / jnp.sum(args.gamma**jnp.arange(args.horizon)),axis=-1)


                    cost_control = ((U_prime - U).reshape(1,1,-1) @ jnp.linalg.inv(cov) @ (V).reshape(args.num_traj, -1,1)).ravel()

                    cost_MPPI = args.alpha1*cost_trajectory + args.alpha2*cost_collision_r2t + args.alpha3 * cost_collision_r2r * args.temperature * (1-args.alpha4) * cost_control + args.alpha5*cost_speed


                    weights = weight_fn(cost_MPPI)


                    if jnp.isnan(cost_MPPI).any():
                        print("BREAK!")
                        break

                    if (mppi_iter < (args.MPPI_iterations-1)): #and (jnp.sum(cost_MPPI*weights) < best_cost):

                        best_cost = jnp.sum(cost_MPPI*weights)

                        U_copy = deepcopy(U_prime)
                        U_prime = U_prime + jnp.sum(weights.reshape(args.num_traj,1,1,1) * E.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)

                        oas = OAS(assume_centered=True).fit(E[weights != 0])
                        cov_prime = jnp.array(oas.covariance_)
                        if mppi_iter == 0:
                            print("Oracle Approx Shrinkage: ",np.round(oas.shrinkage_,5))

                mppi_round_time_end = time()

                if jnp.isnan(cost_MPPI).any():
                    print("BREAK!")
                    break

                weights = weight_fn(cost_MPPI)

                mean_shift = (U_prime - U)

                E_prime = E + mean_shift.ravel()

                U += jnp.sum(weights.reshape(-1,1,1,1) * E_prime.reshape(args.num_traj,args.N_radar,args.horizon,2),axis=0)

                U = jnp.stack((jnp.clip(U[:,:,0],control_constraints[0,0],control_constraints[1,0]),jnp.clip(U[:,:,1],control_constraints[0,1],control_constraints[1,1])),axis=-1)

                # jnp.repeat(U,update_freq_control,axis=1)

                # radar_states = kinematic_model(U ,radar_state, dt_control)

                # generate radar states at measurement frequency
                radar_states = kinematic_model(np.repeat(U, update_freq_control, axis=1)[:, :update_freq_control, :],
                                               radar_state, args.dt_ckf)

                # U += jnp.clip(jnp.sum(weights.reshape(args.num_traj,1,1,1) *  E.reshape(args.num_traj,N,horizon,2),axis=0),U_lower,U_upper)

                # radar_state = radar_states[:,1]
                U = jnp.roll(U, -1, axis=1)

                mppi_end_time = time()
                print(f"MPPI Round Time {step} ",np.round(mppi_end_time-mppi_start_time,3))



        if step >= update_freq_control:
            # get the radar state
            # print("Here",step,(step%update_freq_control)+1)
            if args.move_radars:
                radar_state = radar_states[:,(step%update_freq_control)+1,:]
            else:
                radar_state = radar_state

        J = IM_fn_update(radar_state=radar_state, target_state=ckf.x.reshape(M_target,dm),
                  J=J)

        # print(jnp.linalg.slogdet(J)[1].ravel().item())
        FIMs[step // update_freq_control - 1] = jnp.linalg.slogdet(J)[1].ravel().item()

        if ((step % args.frame_skip) == 0) and ((step % update_freq_control) == 0) and args.save_images:
            print(f"Step {step} - Saving Figure ")

            axes_main[0].plot(radar_state_init[:, 0], radar_state_init[:, 1], 'mo',
                     label="Radar Initial Position")

            thetas = jnp.arcsin(target_state_true[:, 2] / args.R2T)
            radius_projected = args.R2T * jnp.cos(thetas)

            print("Target Height :",target_state_true[:,2])
            print("CKF Target Height :",ckf.x.reshape(M_target,dm)[:,2])
            print("Radius Projected: ",radius_projected)

            radar_state_history[step // update_freq_control - 1] = radar_state

            try:
                imgs_main.append(visualize_tracking(target_state_true=target_state_true, target_state_ckf=ckf.x.reshape(M_target,dm),target_states_true=target_states_true.T.reshape(-1,M_target,dm)[:step],
                           radar_state=radar_state,radar_states_MPPI=radar_states_MPPI,radar_state_history=radar_state_history[max(step // update_freq_control - args.tail_length,0):step // update_freq_control],
                           cost_MPPI=cost_MPPI, FIMs=FIMs[:(step//update_freq_control)],
                           R2T=args.R2T, R2R=args.R2R,C=C,
                           fig=fig_main, axes=axes_main, step=step,
                           tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_CKF"))
            except Exception as error:
                print("Tracking Img Could Not save: ",error)

            try:
                imgs_control.append(visualize_control(U=jnp.roll(U,1,axis=1),CONTROL_LIM=control_constraints,
                           fig=fig_control, axes=axes_control, step=step,
                           tmp_photo_dir = args.tmp_img_savepath, filename = "MPPI_control"))
            except:
                print("Control Img Could Not Save")


        # J = IM_fn(radar_state=radar_state,target_state=m0,J=J)

        # CKF ! ! ! !
        measurement_next_expected = measurement_model(ckf.x.ravel(), radar_state[:,:3], M_target, dm,args.N_radar)

        R = np.diag(C * (measurement_next_expected/2) ** 4)
        ckf.R = R

        range_actual = measurement_model(target_states_true[:, step-1], radar_state[:, :3], M_target, dm, args.N_radar)

        measurement_actual = range_actual + np.random.randn() * np.sqrt(C * (range_actual.ravel() / 2) ** 4)

        ckf.update(np.reshape(measurement_actual,(-1,1)), hx_args=(radar_state[:,:3], M_target, dm,args.N_radar))
        print(f"Step {step} - Tracking ")

        target_state_mse[step-1] = jnp.sqrt(jnp.sum(ckf.x - target_state_true.reshape(-1,1))**2)


    np.savetxt(os.path.join(args.results_savepath,f'rmse_{args.seed}.csv'), np.c_[np.arange(1,args.N_steps+1),target_state_mse], delimiter=',',header="k,rmse",comments='')

    if args.save_images:
        visualize_target_mse(target_state_mse,fig_mse,axes_mse,args.results_savepath,filename="target_mse")

        images = [imageio.imread(file) for file in imgs_main]
        imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_MPC_AIS={args.AIS_method}_FIM={args.fim_method}.gif'), images, duration=0.1)

        images = [imageio.imread(file) for file in imgs_control]
        imageio.mimsave(os.path.join(args.results_savepath, f'MPPI_Control_AIS={args.AIS_method}.gif'), images, duration=0.1)


        if args.remove_tmp_images:
            shutil.rmtree(args.tmp_img_savepath)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Optimal Radar Placement', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # =========================== Experiment Choice ================== #
    parser.add_argument('--seed',default=123,type=int, help='Random seed to kickstart all randomness')
    parser.add_argument('--frame_skip',default=4,type=int, help='Save the images at every nth frame (must be a multiple of the update on the control frequency, which is dt control / dt ckf)')
    parser.add_argument('--dt_ckf', default=0.025,type=float, help='Frequency at which the radar receives measurements and updated Cubature Kalman Filter')
    parser.add_argument('--dt_control', default=0.1,type=float,help='Frequency at which the control optimization problem occurs with MPPI')
    parser.add_argument('--N_radar',default=6,type=int,help="The number of radars in the experiment")
    parser.add_argument("--N_steps",default=1000,type=int,help="The number of steps in the experiment. Total real time duration of experiment is N_steps x dt_ckf")
    parser.add_argument('--results_savepath', default="results",type=str, help='Folder to save bigger results folder')
    parser.add_argument('--experiment_name', default="experiment",type=str, help='Name of folder to save temporary images to make GIFs')
    parser.add_argument('--move_radars', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to allow the radars to move? --move_radars for yes --no-move_radars for no')
    parser.add_argument('--remove_tmp_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to remove tmp images? --remove_tmp_images for yes --no-remove_tmp_images for no')
    parser.add_argument('--tail_length',default=10,type=int,help="The length of the tail of the radar trajectories in plottings")
    parser.add_argument('--save_images', action=argparse.BooleanOptionalAction,default=True,help='Do you wish to saves images/gifs? --save_images for yes --no-save_images for no')
    parser.add_argument('--fim_method', default="Standard_FIM",type=str, help='FIM Calculation [SFIM,PFIM]')

    # ==================== RADAR CONFIGURATION ======================== #
    parser.add_argument('--fc', default=1e8,type=float, help='Radar Signal Carrier Frequency (Hz)')
    parser.add_argument('--Gt', default=200,type=float, help='Radar Transmit Gain')
    parser.add_argument('--Gr', default=200,type=float, help='Radar Receive Gain')
    parser.add_argument('--rcs', default=1,type=float, help='Radar Cross Section in m^2')
    parser.add_argument('--L', default=1,type=float, help='Radar Loss')
    parser.add_argument('--R', default=500,type=float, help='Radius for specific SNR (desired)')
    parser.add_argument('--Pt', default=1000,type=float, help='Radar Power Transmitted (W)')
    parser.add_argument('--SNR', default=-20,type=float, help='Signal-Noise Ratio for Experiment. Radar has SNR at range R')


    # ==================== MPPI CONFIGURATION ======================== #
    parser.add_argument('--acc_std', default=25,type=float, help='Radar Signal Carrier Frequency (Hz)')
    parser.add_argument('--ang_acc_std', default=45*jnp.pi/180,type=float, help='Radar Transmit Gain')
    parser.add_argument('--horizon', default=15,type=int, help='Radar Receive Gain')
    parser.add_argument('--acc_init', default=0,type=float, help='Radar Cross Section in m^2')
    parser.add_argument('--ang_acc_init', default= 0 * jnp.pi/180,type=float, help='Radar Loss')
    parser.add_argument('--num_traj', default=250,type=int, help='Number of MPPI control sequences samples to generate')
    parser.add_argument('--MPPI_iterations', default=25,type=int, help='Number of MPPI sub iterations (proposal adaptations)')

    # ==================== AIS  CONFIGURATION ======================== #
    parser.add_argument('--temperature', default=0.1,type=float, help='Temperature on the objective function. Lower temperature accentuates the differences between scores in MPPI')
    parser.add_argument('--elite_threshold', default=0.9,type=float, help='Elite Threshold (between 0-1, where closer to 1 means reject most samaples)')
    parser.add_argument('--AIS_method', default="CE",type=str, help='Type of importance sampling. [CE,information]')

    # ============================ MPC Settings =====================================#
    parser.add_argument('--gamma', default=0.95,type=float, help="Discount Factor for MPC objective")
    parser.add_argument('--speed_minimum', default=5,type=float, help='Minimum speed Radars should move [m/s]')
    parser.add_argument('--R2T', default=125,type=float, help='Radius from Radar to Target to maintain [m]')
    parser.add_argument('--R2R', default=10,type=float, help='Radius from Radar to  Radar to maintain [m]')
    parser.add_argument('--alpha1', default=1,type=float, help='Cost weighting for FIM')
    parser.add_argument('--alpha2', default=1000,type=float, help='Cost weighting for maintaining distanace between Radar to Target')
    parser.add_argument('--alpha3', default=60,type=float, help='Cost weighting for maintaining distance between Radar to Radar')
    parser.add_argument('--alpha4', default=1,type=float, help='Cost weighting for smooth controls (between 0 to 1, where closer to 1 means no smoothness')
    parser.add_argument('--alpha5', default=0,type=float, help='Cost weighting to maintain minimum absolute speed')


    args = parser.parse_args()


    args.results_savepath = os.path.join(args.results_savepath,args.experiment_name) + f"_{args.seed}"
    args.tmp_img_savepath = os.path.join( args.results_savepath,"tmp_img") #('--tmp_img_savepath', default=os.path.join("results","tmp_images"),type=str, help='Folder to save temporary images to make GIFs')



    from datetime import datetime
    from pytz import timezone
    import json

    tz = timezone('EST')
    print("Experiment State @ ",datetime.now(tz))
    print("Experiment Saved @ ",args.results_savepath)
    print("Experiment Settings Saved @ ",args.results_savepath)


    os.makedirs(args.tmp_img_savepath,exist_ok=True)
    os.makedirs(args.results_savepath,exist_ok=True)

    # Convert and write JSON object to file
    with open(os.path.join(args.results_savepath,"hyperparameters.json"), "w") as outfile:
        json.dump(vars(args), outfile)

    main(args)
