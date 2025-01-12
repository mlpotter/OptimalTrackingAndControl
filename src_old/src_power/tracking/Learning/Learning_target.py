# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:38:21 2024

@author: siliconsynapse
"""

from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import vmap
import functools
from jax import random

from src_range.utils import NoiseParams,place_sensors
from src_range.tracking.Particle_Filter import *

import numpy as np

from tqdm import tqdm
import os,glob

import matplotlib.pyplot as plt
import imageio


def create_frame(t, Xnext, XT, ps, m0, ws, neffs, vs, heights, axes=None):
    wmin, wmax = jnp.min(ws), jnp.max(ws)
    ws_minmax = (ws - wmin) / (wmax - wmin + 1e-16)
    pf_point = (Wnext * Xnext).sum(axis=0).ravel()
    pf_point = pf_point.reshape(M, d)
    Xnext = Xnext.reshape(-1, M, d)
    XT = XT.reshape(-1, M, d)

    pfs = axes[0].scatter(Xnext[:, :, 0], Xnext[:, :, 1], c='green', alpha=ws_minmax, label="Particle States")
    target_traj = axes[0].plot(XT[:, :, 0], XT[:, :, 1], marker='d', color="r", alpha=0.3, label='_nolegend_')
    empty_target_traj = axes[0].plot([], [], marker='d', color="r", linestyle='', label="Target")
    sensor_pos = axes[0].plot(ps[:, 0], ps[:, 1], marker="o", color="b", linestyle='', label='Sensor')
    pf_init = axes[0].plot(m0[:, 0], m0[:, 1], marker="x", color="g", linestyle='', label="Init Guess")
    pf_mean = axes[0].plot(pf_point[:, 0], pf_point[:, 1], color="k", marker='x', linestyle='', label="PF Avg")
    axes[0].set_ylabel("$y$");
    axes[0].set_xlabel("$x$");
    axes[0].legend()
    axes[0].set_title(f"t={t + 1}")

    ws_top50 = jnp.sort(ws.ravel())[::-1]  # [-100:][::-1]
    stems = axes[1].stem(ws_top50)
    axes[1].set_ylabel("Particle Weight")
    axes[1].set_title(f"{len(ws_top50)} PF Weights")

    neffs = axes[2].plot(neffs, 'ko-')
    axes[2].set_ylabel("Neff")
    axes[2].set_xlabel("t")

    neffs = axes[3].plot(vs, 'o-')
    axes[3].set_ylabel("Radar Measure")
    axes[3].set_xlabel("t")

    height = axes[4].plot(heights, marker='o', color="y", label='_nolegend_', linestyle="-")
    axes[4].set_ylabel("Height Estimates")
    axes[4].set_xlabel("t")
    # axes[3].legend([f"radar={n}" for n in range(N)])

    filename = f"frame_{t}.png"
    plt.savefig(os.path.join(photo_dump, filename))

    axes[0].legend_ = None
    # axes[3].legend_=None
    pfs.remove()
    [line.remove() for line in target_traj]
    [line.remove() for line in sensor_pos]
    [line.remove() for line in pf_init]
    [line.remove() for line in pf_mean]
    [line.remove() for line in empty_target_traj]

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()
    axes[3].cla()
    axes[4].cla()

    return os.path.join(photo_dump, filename)


if __name__ == "__main__":
    seed = 123
    key = jax.random.PRNGKey(seed)

    key = random.PRNGKey(0)
    np.random.seed(123)

    speedoflight = 299792458
    fc = 1e9;
    Gt = 2000;
    Gr = 2000;
    lam = speedoflight / fc
    rcs = 1;
    L = 1

    # calculate Pt such that I achieve SNR=x at distance R=y
    R = 100

    SCNR = -20
    CNR = -10

    coef = Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3 / (R ** 4)
    Pt = 10000
    _, _, _, s = NoiseParams(Pt * coef, SCNR, CNR=CNR)

    print("Transmit Power: ", Pt)
    print("Radar Return (RCS)",coef*Pt)
    print("Spread: ", s ** 2)

    # Generic experiment
    T = 0.01
    NP = 5000
    TN = 1000
    N = 4
    photo_dump = os.path.join("tmp_images")
    gif_filename = f"pf_track_{SCNR}.gif"
    gif_savepath = os.path.join("..", "..", "images")
    remove_photo_dump = True
    every_other_frame = 25
    os.makedirs(photo_dump, exist_ok=True)


    ps = place_sensors([-100, 100], [-100, 100], N)

    z_elevation = 150
    qs = jnp.array([[0.0, -0.0, z_elevation, 50., 20.2, 0],
                    [-3.1, 2.23, z_elevation, .1, 15.0, 0]])  # ,
    # [-5.4,3.32,z_elevation,-5,-5,0]])

    M, d = qs.shape;
    N = len(ps);

    sigmaQ = np.sqrt(10 ** (3));

    print("SigmaQ (state noise)={}".format(sigmaQ))

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
    key, XT, YT = generate_samples(key, qs, ps, A, Q,
                                   Gt,Gr,Pt,lam,rcs,L,s,
                                   TN)

    key, subkey = random.split(key)
    m0 = qs.at[:, 3:].add(5);
    m0 = m0.at[:, :3].add(-20)

    P0_singular = jnp.diag(jnp.array([50, 50, 50, 50, 50, 50]));
    P0 = jnp.kron(jnp.eye(M), P0_singular)

    Xprev = jax.random.multivariate_normal(subkey, m0.ravel(), P0, shape=(NP,), method="cholesky")
    Wprev = jnp.ones((NP, 1)) * 1 / NP

    key, Xnext = optimal_importance_dist_sample(key, Xprev, A, Q)

    Filter = np.zeros((TN, nx))

    Ws = [];
    Filters = [];
    Measures = [];
    RMSE = [];
    Neffs = []

    frame_names = []
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5));
    Vs = jnp.zeros((TN, N * M))
    heights = jnp.zeros((TN, M))
    Vnext_parallel = vmap(functools.partial(RadarEqnMeasure,Gt=Gt,Gr=Gr,Pt=Pt,lam=lam,rcs=rcs,L=L), in_axes=(0, None))
    for t in tqdm(range(0, TN)):
        # get the measurement  y_k+1
        ynext = YT[t].reshape(-1, 1)

        # sample with state transition matrix x_k+1
        Xnext, Wnext= Rnode_Predict()
        
        key, Xnext = optimal_importance_dist_sample(key, Xprev, A, Q)
        heights = heights.at[t].set(Xnext.reshape(-1, M, d).mean(axis=0)[:, 2])

        # Generate the expected Measurements for x_k+1 (v_k+1)
        Vnext = Vnext_parallel(Xnext.reshape(-1, M, nx // M), ps)
        Vs = Vs.at[t].set(Vnext.mean(0).ravel())

        # Update particle weights recursively...
        Xnext=RNODE_state_update()
        W_next=RNODE_weight_update()
        Wnext = weight_update(Wprev, Vnext, s,ynext)
        neffs = effective_samples(Wnext)
        Measures.append(Vnext)
        Neffs.append(neffs)

        # if np.isnan(Wnext).any():
        #     Wnext = jnp.ones((NP, 1)) * 1 / NP
        if jnp.isnan(Wnext).any().item():
            print("Particle Filter Failed!")
            break

        if t % every_other_frame == 0:
            frame = create_frame(t, Xnext, XT[:t + 1], ps, m0, Wnext, Neffs, YT[:t + 1], heights[:t + 1], axes=axes)
            frame_names.append(frame)

        if neffs < (NP * 2 / 3) or (t + 1) % 250 == 0:
            print("\n Particle Resampling")
            Xnext, Wnext = weight_resample(Xnext, Wnext)

        Filter[t] = (Wnext * Xnext).sum(axis=0)

        Ws.append(Wnext);
        Filters.append(Xnext)
        RMSE.append(jnp.sqrt((Filter[t] - XT[t]) ** 2))

        Xprev = Xnext;
        Wprev = Wnext

        print(f"\n Effective Particle Count: {neffs} , Particle Weight Variance: {np.var(Wprev.ravel())}")

    # Save frames as a GIF
    frames = []
    for frame_name in frame_names:
        frames.append(imageio.imread(frame_name))

    imageio.mimsave(os.path.join(gif_savepath, "gifs", gif_filename), frames,
                    duration=0.25)  # Adjust duration as needed
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
    plt.savefig(os.path.join(gif_savepath, f"pf_rmse_{SCNR}.png"))
    plt.close()

    # # Dummy plots for creating the legend
    # dummy_plot1, = plt.plot([], [], 'b-', label='Constant Velocity')
    # dummy_plot2, = plt.plot([], [], 'r--', label='Particle Filter')

    # plt.plot(XT.reshape(-1, M, nx // M)[:, :, 0], XT.reshape(-1, M, nx // M)[:, :, 1], 'b-')
    # plt.plot(Filter.reshape(-1, M, nx // M)[:, :, 0], Filter.reshape(-1, M, nx // M)[:, :, 1], 'r--')
    # plt.legend(handles=[dummy_plot1, dummy_plot2])