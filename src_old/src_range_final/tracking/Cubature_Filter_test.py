from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import vmap
import functools
from jax import random

from src_range_final.utils import place_sensors
from src_range.tracking.Particle_Filter import generate_range_samples,optimal_importance_dist_sample,weight_update,effective_samples,weight_resample
from src_range_final.tracking.Cubature_Filter import ckf_predict,ckf_filter,generate_sigma_points
from src_range_final.tracking.Measurement_Models import RangeVelocityMeasure
from src_range_final.tracking import Particle_Filter
import numpy as np

from tqdm import tqdm
import os,glob

import matplotlib.pyplot as plt
import imageio


def create_frame(t, XT, ps, target_state_filter, target_measurement_actual, heights, sigma_points,axes=None):

    XT = XT.reshape(-1, M, dm)
    sigma_points = sigma_points.T.reshape(-1,M,dm)

    target_traj = axes[0].plot(XT[:, :, 0], XT[:, :, 1], marker='d', color="r", alpha=0.3, label='_nolegend_')
    empty_target_traj = axes[0].plot([], [], marker='d', color="r", linestyle='', label="Target")
    sensor_pos = axes[0].plot(ps[:, 0], ps[:, 1], marker="o", color="b", linestyle='', label='Sensor')

    sigma_pts = axes[0].plot(sigma_points[:,:, 0], sigma_points[:,:, 1], color="g", marker='x', linestyle='', label="Sigma Points")
    pf_mean = axes[0].plot(target_state_filter[:, 0], target_state_filter[:, 1], color="k", marker='x', linestyle='', label="CF Avg")

    axes[0].set_ylabel("$y$");
    axes[0].set_xlabel("$x$");
    axes[0].legend()
    axes[0].set_title(f"t={t + 1}")

    ranges = axes[1].plot(target_measurement_actual, 'o-')
    axes[1].set_ylabel("Range Measure")
    axes[1].set_xlabel("t")

    height = axes[2].plot(heights, marker='o', color="y", label='_nolegend_', linestyle="-")
    axes[2].set_ylabel("Height Estimates")
    axes[2].set_xlabel("t")
    # axes[3].legend([f"radar={n}" for n in range(N)])

    filename = f"frame_{t}.png"
    plt.savefig(os.path.join(photo_dump, filename))

    axes[0].legend_ = None
    # axes[3].legend_=None
    [line.remove() for line in target_traj]
    [line.remove() for line in sensor_pos]
    [line.remove() for line in empty_target_traj]

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()

    return os.path.join(photo_dump, filename)


if __name__ == "__main__":
    seed = 123
    key = jax.random.PRNGKey(seed)

    np.random.seed(123)

    c = 299792458
    fc = 1e6;
    Gt = 2000;
    Gr = 2000;
    lam = c / fc
    rcs = 1;
    L = 1;
    # alpha = (jnp.pi)**2 / 3
    # B = 0.05 * 10**5

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 1000

    Pt = 10000
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
    Pr = K / (R ** 4)
    # get the power of the noise of the signal
    SNR=-0


    # Generic experiment
    T = 0.1
    TN = 500
    N = 4
    photo_dump = os.path.join("tmp_images")
    gif_filename = f"cf_track_{SNR}.gif"
    gif_savepath = os.path.join("..", "..", "images")
    remove_photo_dump = False
    every_other_frame = 25
    os.makedirs(photo_dump, exist_ok=True)


    ps = place_sensors([-500, 500], [-500, 500], N)

    # z_elevation = 150
    # qs = jnp.array([[0.0, -0.0, 50., 20.2],
    #                 [-3.1, 2.23, .1, 15.0]])  # ,
    # [-5.4,3.32,z_elevation,-5,-5,0]])
    z_elevation = 150
    qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0], #,#,
                    [-50.4,30.32,z_elevation,-20,-10,0]])#, #,
                    # [10,10,z_elevation,10,10,0],
                    # [20,20,z_elevation,5,-5,0]])

    M, dm = qs.shape;
    N , dn = ps.shape

    sigmaQ = np.sqrt(10 ** (3));
    sigmaV = np.sqrt(5)
    sigmaW = jnp.sqrt(M*Pr/ (10**(SNR/10)))
    C = c**2 * sigmaW**2 / (fc**2 * 8 * jnp.pi**2) * 1/K

    print("SigmaQ (state noise)={}".format(sigmaQ))
    print("Transmit Power: ", Pt)
    print("Radar Return (RCS)",Pr)
    print("Noise Power: ", sigmaW**2)
    print("Scaling Factor: ",C)

    A_single = jnp.array([[1., 0, 0, T, 0, 0],
                          [0, 1., 0, 0, T, 0],
                          [0, 0, 1, 0, 0, T],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1., 0],
                          [0, 0, 0, 0, 0, 1]])

    Q_single = jnp.array([
        [(T ** 4) / 4, 0, 0, (T ** 3) / 2, 0, 0],
        [0, (T ** 4) / 4, 0, 0, (T ** 3) / 2, 0],
        [0, 0, (T ** 4) / 4, 0, 0, (T ** 3) / 2],
        [(T ** 3) / 2, 0, 0, (T ** 2), 0, 0],
        [0, (T ** 3) / 2, 0, 0, (T ** 2), 0],
        [0, 0, (T ** 3) / 2, 0, 0, (T ** 2)]
    ]) * sigmaQ ** 2
    #
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

    A = jnp.kron(jnp.eye(M), A_single);
    Q = jnp.kron(jnp.eye(M), Q_single);  # + np.eye(M*Q_single.shape[0])*1e-1;
    G = jnp.eye(N)

    nx = Q.shape[0]

    # Generate the trajectory and measurements to trajectory: X_1:k, Y_1:k
    key, XT, YT = generate_range_samples(key, qs, ps, A=A, Q=Q,
                                   Gt=Gt,Gr=Gr,Pt=Pt,lam=lam,rcs=rcs,L=L,c=c,fc=fc,sigmaW=sigmaW,sigmaV=sigmaV,
                                   TN=TN)

    key, subkey = random.split(key)
    target_state_filter = qs.at[:, dm//2:].add(3);
    target_state_filter = target_state_filter.at[:, :dm//2].add(5);

    P_filter = jnp.diag(jnp.array([25, 25, 25, 25, 25, 25]));
    # P0_singular = jnp.diag(jnp.array([50, 50, 50, 50]));

    P_filter  = jnp.kron(jnp.eye(M), P_filter)


    Filter = np.zeros((TN, nx))

    Measures = [];
    RMSE = [];

    frame_names = []
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5));
    Vs = jnp.zeros((TN, N * M * (dm//2 + 1)))
    heights = jnp.zeros((TN, M))

    # weight_update_partial = partial(weight_update,Pt=Pt,Gt=Gt,Gr=Gr,lam=lam,rcs=rcs,L=L,c=c,fc=fc,sigmaW=sigmaW,sigmaV=sigmaV,M=M,N=N,dm=dm,dn=dn)
    for t in tqdm(range(0, TN)):
        # predict the next state
        target_state_predict,P_predict = ckf_predict(target_state_filter, P_filter, A, Q)

        # predict sigma points
        sigma_points = generate_sigma_points(target_state_predict, P_predict, M*dm)

        # get the measurement  y_k+1
        target_measurement_actual = YT[t].reshape(-1,1)

        # get the expected measurement y_k+1
        target_measurement_expected = RangeVelocityMeasure(target_state_predict[:,:dm//2],ps)

        # get the covariance matrix based on the next measurement
        range_measures = target_measurement_expected[:, 0].ravel()

        range_var = C * (range_measures ** 4)

        variances = jnp.column_stack((range_var,jnp.ones((range_measures.shape[0],dm//2))*sigmaV**2))

        cov_measurement = jnp.diag(variances.ravel())


        target_state_filter,P_filter = ckf_filter(target_measurement_actual.reshape(-1,1),ps,target_state_predict,P_predict,cov_measurement)
        # if t % 25 == 0 :
        #     print("Target State: ",target_state_filter)
        #     print("True State: ",XT[t].reshape(M,dm))

        heights = heights.at[t].set(target_state_filter.reshape(M, dm)[:, 2])

        # if np.isnan(Wnext).any():
        #     Wnext = jnp.ones((NP, 1)) * 1 / NP

        if t % every_other_frame == 0:
            frame = create_frame(t, XT[t], ps, target_state_predict, YT[:t + 1,:,0],  heights[:t + 1], sigma_points,axes=axes)
            frame_names.append(frame)


        # Filters.append(target_state_filter)
        RMSE.append(jnp.sqrt((target_state_filter.ravel()- XT[t]) ** 2))

    # Save frames as a GIF
    frames = []
    for frame_name in frame_names:
        frames.append(imageio.imread(frame_name))

    imageio.mimsave(os.path.join(gif_savepath, "gifs", gif_filename), frames,
                    duration=1000)  # Adjust duration as needed
    print(f"GIF saved as '{gif_filename}'")

    if remove_photo_dump:
        for filename in glob.glob(os.path.join(photo_dump, "frame_*")):
            os.remove(filename)

    RMSE = jnp.stack(RMSE, axis=0)
    plt.figure()
    plt.plot(RMSE)
    # plt.legend(["$x$","$y$","$v_x$","$v_y$"])
    plt.title("States")
    plt.ylabel("RMSE")
    plt.xlabel("time step")
    plt.savefig(os.path.join(gif_savepath, f"cf_rmse_{SNR}.png"))
    plt.close()

    # # Dummy plots for creating the legend
    # dummy_plot1, = plt.plot([], [], 'b-', label='Constant Velocity')
    # dummy_plot2, = plt.plot([], [], 'r--', label='Particle Filter')

    # plt.plot(XT.reshape(-1, M, nx // M)[:, :, 0], XT.reshape(-1, M, nx // M)[:, :, 1], 'b-')
    # plt.plot(Filter.reshape(-1, M, nx // M)[:, :, 0], Filter.reshape(-1, M, nx // M)[:, :, 1], 'r--')
    # plt.legend(handles=[dummy_plot1, dummy_plot2])