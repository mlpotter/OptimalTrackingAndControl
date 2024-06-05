from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jaxopt import ScipyMinimize



from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,Single_FIM_Radar,Multi_FIM_Logdet_decorator_MPC,FIM_Visualization

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

from src_range.utils import NoiseParams
from src_range.control.MPPI import *

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
    NT = 115
    MPPI_FLAG = True
    PRUNE_FLAG = False
    MPPI_VISUALIZE = False
    MPPI_ITER_VISUALIZE = True
    fim_method = "Standard FIM"

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
    time_steps = 20
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
    v_init = 0
    av_init = 0

    num_traj = 250
    MPPI_iterations = 50
    MPPI_method = "single"
    method = "Single_FIM_3D_action_MPPI"
    u_ptb_method = "mixture"

    # ==================== AIS CONFIGURATION ================================= #
    temperature = 0.1
    alpha = 0.8
    elite_threshold = 0.9
    AIS_method = "information"

    gif_savename =  f"JU_MPPI2_{AIS_method}.gif"


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
    qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
                    [-100.4,-30.32,z_elevation,20,-10,0], #,
                    [30,30,z_elevation,-10,-10,0],
                    [-20,60,z_elevation,-15,15,0]])
    # qs = jnp.array([[0.0, -0.0,z_elevation, 0., 0,0], #,#,
    #                 [-50.4,30.32,z_elevation,-0,-0,0], #,
    #                 [10,10,z_elevation,0,0,0],
    #                 [20,20,z_elevation,0,0,0]])

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
    sigmaV = jnp.sqrt(9)

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


    IM_fn(ps,qs,J=J)

    # IM_fn_parallel = vmap(IM_fn, in_axes=(None, 0, 0))

    Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC(IM_fn=IM_fn,method=method)

    MPPI_scores = MPPI_scores_wrapper(Multi_FIM_Logdet,method=MPPI_method)

    if AIS_method == "CE":
        weight_fn = partial(weighting(AIS_method),elite_threshold=elite_threshold)
    elif AIS_method == "information":
        weight_fn = partial(weighting(AIS_method),temperature=temperature)

    chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
    # time_step_sizes = jnp.tile(time_step_size, (N, 1))

    U_upper = (jnp.ones((time_steps, 2)) * jnp.array([[max_velocity, max_angle_velocity]]))
    U_lower = (jnp.ones((time_steps, 2)) * jnp.array([[min_velocity, min_angle_velocity]]))

    U_lower = jnp.tile(U_lower, jnp.array([N, 1, 1]))
    U_upper = jnp.tile(U_upper, jnp.array([N, 1, 1]))

    unicycle_kinematics_vmap = vmap(unicycle_kinematics, (0, 0, 0, None))

    m0 = qs
    images = []; images_mppi = []
    FIMs = []
    qs_previous = None

    U_V = jnp.ones((N,time_steps,1)) * v_init
    U_W = jnp.ones((N,time_steps,1)) * av_init
    U_Nom =jnp.concatenate((U_V,U_W),axis=-1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fig_debug,axes_debug = plt.subplots(1,1,figsize=(10,10))

    for k in range(NT):
        print(f"\n Step {k} MPPI Iteration: ")
        qs_previous = m0
        m0 = (A @ m0.reshape(-1, 1)).reshape(M, dm)

        best_mppi_iter_score = np.inf
        mppi_round_time_start = time()

        target_states_rollout = jnp.stack([(jnp.linalg.matrix_power(A,t-1) @ m0.reshape(-1, M * dm).T).T.reshape(M, dm) for t in range(1,time_steps+1)])

        for mppi_iter in range(MPPI_iterations):
            start = time()
            key, subkey = jax.random.split(key)

            mppi_start = time()
            U_ptb = MPPI_ptb(stds,N, time_steps, num_traj, key,method=u_ptb_method)

            mppi_rollout_start = time()
            U_MPPI,P_MPPI,CHI_MPPI, _,_,_ = MPPI(U_nominal=U_Nom, chis_nominal=chis,
                                                               U_ptb=U_ptb,ps=ps,
                                                               time_step_size=time_step_size, limits=limits)
            mppi_rollout_end = time()
            # if PRUNE_FLAG:
            #     differences = P_MPPI.squeeze(3)[:,:,jnp.newaxis] - m0[jnp.newaxis, jnp.newaxis, :,jnp.newaxis, :2]
            #     distances = jnp.sqrt(jnp.sum((differences ** 2), -1))
            #     prune = (distances<R_sensors_to_targets)
            #     prune = np.logical_not(prune.any(axis=[1, 2, 3]))
            #
            #     print("Kept: ",prune.sum())
            #
            #     U_MPPI = U_MPPI[prune]
            #     P_MPPI = P_MPPI[prune]
            #     CHI_MPPI = CHI_MPPI[prune]
            #
            #      # plt.plot(P_MPPI.squeeze()[:,0,:,0].T,P_MPPI.squeeze()[:,0,:,1].T)
            mppi_score_start = time()
            cost_MPPI = MPPI_scores(ps, target_states_rollout, U_MPPI,chis, time_step_size,
                                      A=A,J=J,
                                      gamma=gamma)
            mppi_score_end = time()
            # print("MPPI Score Time: ", mppi_score_end - mppi_start)


            min_idx = jnp.argmin(cost_MPPI)
            lowest_cost = cost_MPPI[min_idx]

            if lowest_cost < best_mppi_iter_score:
                if k == 0:
                    print(f"First Iter Best Score {mppi_iter}: ",-lowest_cost)
                best_mppi_iter_score = lowest_cost
                U_BEST = U_MPPI[min_idx]

                # print(SCORE_BEST)

            # scores_MPPI_weight = jax.nn.softmax(scores_MPPI)

            scores_MPPI_weight = weight_fn(cost_MPPI)

            delta_actions = U_MPPI - U_Nom
            # U_Nom = jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1), axis=0)
            U_Nom += jnp.sum(delta_actions * scores_MPPI_weight.reshape(-1, 1, 1, 1), axis=0)

            if k==0 and MPPI_ITER_VISUALIZE:
                file_mppi = os.path.join("tmp_images", f"MPPI_single_iteration_{mppi_iter}.png")
                images_mppi.append(file_mppi)
                axes_debug.plot(qs_previous[:,0], qs_previous[:,1], 'g.',label="Target Init Position")
                axes_debug.plot(m0[:,0], m0[:,1], 'go',label="Target Position")
                _, _, Sensor_Positions_debug, Sensor_Chis_debug = vmap(unicycle_kinematics, (0, 0, 0, None))(jnp.expand_dims(ps, 1), U_Nom,chis, time_step_size)
                _, _, Sensor_Positions_best_debug, Sensor_Chis_best_debug = vmap(unicycle_kinematics, (0, 0, 0, None))(jnp.expand_dims(ps, 1), U_BEST,chis, time_step_size)

                if MPPI_VISUALIZE:
                    for n in range(N):
                        plt.plot(P_MPPI[:, n, :, 0].T, P_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
                axes_debug.plot(Sensor_Positions_debug[:,0,0], Sensor_Positions_debug[:,0,1], 'r*',label="Sensor Position")
                axes_debug.plot(Sensor_Positions_debug[:,1:,0].T, Sensor_Positions_debug[:,1:,1].T, 'r-',label="_nolegend_")
                axes_debug.plot(Sensor_Positions_best_debug[:,1:,0].T, Sensor_Positions_best_debug[:,1:,1].T, color="lime",linestyle='-',label="_nolegend_")

                fig_debug.tight_layout()
                fig_debug.savefig(file_mppi)
                axes_debug.cla()

            # U_BEST = jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
            mppi_end = time()
            # print("MPPI Mean Score: ",np.mean(scores_MPPI))
            # print("MPPI Best Score: ",np.argmin(scores_MPPI))
            # print("MPPI TIME: ",mppi_end-mppi_start)

        mppi_round_time_end = time()
        U_Nom = jnp.roll(U_BEST,-1,axis=1)
        print("MPPI Round Time: ",mppi_round_time_end-mppi_round_time_start)
        print("MPPI Iter Time: ",mppi_end-mppi_start)
        print("MPPI Score Time: ",mppi_score_end-mppi_score_start)
        print("MPPI Mean Score: ",jnp.nanmean(-cost_MPPI))
        print("MPPI Best Score: ",-best_mppi_iter_score)
        # FIMs.append(-jnp.nanmean(scores_MPPI))



        # U_BEST =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
        # U_nominal =  jnp.sum(U_MPPI * scores_MPPI_weight.reshape(-1, 1, 1, 1),axis=0)
        _, _, Sensor_Positions, Sensor_Chis = unicycle_kinematics_vmap(jnp.expand_dims(ps, 1), U_BEST ,
                                                                       chis, time_step_size)

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
        imageio.mimsave(os.path.join(gif_savepath,f'MPPI_step_{AIS_method}.gif'),images,duration=0.1)#