import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from FIM_new.FIM_RADAR import FIM_Visualization
import numpy as np
import os

# from gym collision avoidance library
plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980])  # orange
plt_colors.append([0.0, 0.4470, 0.7410])  # blue
plt_colors.append([0.4660, 0.6740, 0.1880])  # green
plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
plt_colors.append([0.9290, 0.6940, 0.1250])  # yellow
plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
plt_colors.append([0.6350, 0.0780, 0.1840])  # red
def rgba2rgb(rgba):
    # rgba is a list of 4 color elements btwn [0.0, 1.0]
    # or a 2d np array (num_colors, 4)
    # returns a list of rgb values between [0.0, 1.0] accounting for alpha and background color [1, 1, 1] == WHITE
    if isinstance(rgba, list):
        alpha = rgba[3]
        r = max(min((1 - alpha) * 1.0 + alpha * rgba[0],1.0),0.0)
        g = max(min((1 - alpha) * 1.0 + alpha * rgba[1],1.0),0.0)
        b = max(min((1 - alpha) * 1.0 + alpha * rgba[2],1.0),0.0)
        return [r,g,b]
    elif rgba.ndim == 2:
        alphas = rgba[:,3]
        r = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,0], 0, 1)
        g = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,1], 0, 1)
        b = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,2], 0, 1)
        return np.vstack([r,g,b]).T

def visualize_target_mse(MSE,fig,axes,tmp_photo_dir,filename="visualize"):
    os.makedirs(tmp_photo_dir,exist_ok=True)
    file_savepth = os.path.join(tmp_photo_dir,filename+f".png")


    axes.plot(MSE.ravel(),'mo-')


    axes.set_title("$RMSE$")
    axes.set_xlabel("Time Step")
    axes.set_ylabel("$\sqrt{|| x_{true} - x_{ckf} ||}$")

    fig.tight_layout()
    fig.savefig(file_savepth)

    return file_savepth
def visualize_control(U,CONTROL_LIM,fig,axes,step,tmp_photo_dir,filename="visualize"):
    os.makedirs(tmp_photo_dir,exist_ok=True)
    file_savepth = os.path.join(tmp_photo_dir,filename+f"_{step}.png")

    N,horizon,dc = U.shape
    colors = plt.cm.jet(np.linspace(0, 1, N))

    time = np.tile(np.arange(horizon),(N,1)).reshape(N,horizon,1)

    U1_segs = LineCollection(np.concatenate((time,U[...,[0]]),axis=-1), colors=colors, alpha=0.5)

    U2_segs = LineCollection(np.concatenate((time,U[...,[1]]),axis=-1), colors=colors, alpha=0.5)

    axes[0].add_collection(U1_segs)
    axes[1].add_collection(U2_segs)

    axes[0].set_ylim([CONTROL_LIM[0,0]-np.abs(CONTROL_LIM[0,0])*0.05,CONTROL_LIM[1,0]+np.abs(CONTROL_LIM[1,0]*0.05)])
    axes[1].set_ylim([CONTROL_LIM[0,1]-np.abs(CONTROL_LIM[0,1])*0.05,CONTROL_LIM[1,1]+np.abs(CONTROL_LIM[1,1]*0.05)])
    axes[0].set_xlim([0,horizon-1])
    axes[1].set_xlim([0,horizon-1])

    axes[0].set_title("$U_1$")
    axes[0].set_xlabel("Time Step")
    axes[1].set_title("$U_2$")
    axes[1].set_xlabel("Time Step")

    fig.suptitle(f"Iteration {step}")
    fig.tight_layout()
    fig.savefig(file_savepth)
    axes[0].cla()
    axes[1].cla()

    return file_savepth

def visualize_tracking(target_state_true,target_state_ckf,target_states_true,
                       cost_MPPI,
                       radar_state,radar_states_MPPI,radar_state_history,
                       FIMs,
                       R2T,R2R,C,
                       fig,axes,step,tmp_photo_dir,filename="visualize"):

    os.makedirs(tmp_photo_dir,exist_ok=True)
    file_savepth = os.path.join(tmp_photo_dir,filename+f"_{step}.png")

    # Main figure

    M_target,dm = target_state_true.shape
    N_radar,dn = radar_state.shape

    thetas_ckf = jnp.arcsin(np.abs(target_state_ckf[:, 2]) / R2T)
    radius_projected_ckf = R2T * jnp.cos(thetas_ckf)

    thetas_true = jnp.arcsin(np.abs(target_state_true[:, 2]) / R2T)
    radius_projected_true = R2T * jnp.cos(thetas_true)

    # target_traj_segs =     LineCollection( np.swapaxes(target_states_true[:,:,:2],1,0), colors="g", alpha=0.5)

    colors_target = np.zeros((target_states_true.shape[0], 4))
    colors_target[:, :3] = plt_colors[2]
    colors_target[:, 3] = np.linspace(0.2, 1., target_states_true.shape[0])
    colors_target = rgba2rgb(colors_target)
    axes[0].scatter(target_states_true[...,0].T,target_states_true[...,1].T,color=np.tile(colors_target,(M_target,1)),label="_nolegend_")


    # axes[0].plot(radar_state_history[0, :, 0], radar_state_history[0, :, 1], 'md', label="Sensor Init")
    axes[0].plot(target_state_true[:, 0].ravel(), target_state_true[:, 1].ravel(), 'go', label="Target Position")
    # axes[0].add_collection(target_traj_segs)

    colors_radar = np.zeros((radar_state_history.shape[0], 4))
    colors_radar[:, :3] = plt_colors[-1]
    colors_radar[:, 3] = np.linspace(0.2, 1., radar_state_history.shape[0])
    colors_radar = rgba2rgb(colors_radar)

    axes[0].scatter(radar_state_history[...,0].T,radar_state_history[...,1].T,color=np.tile(colors_radar,(N_radar,1)),label="_nolegend_")
    axes[0].scatter([],[],color=colors_radar[-1],label="Radar Position")
    # axes[0].plot(radar_state[:, 0].ravel(), radar_state[:, 1].ravel(), 'r*', label="Radar")

    axes[0].plot(target_state_ckf[:, 0].ravel(), target_state_ckf[:, 1].ravel(), 'bX', label="CKF Predict Position")

    for m in range(M_target):
        axes[0].add_patch(
            Circle(target_state_true[m, :2], radius_projected_true[m], edgecolor="green", fill=False, lw=1,
                   linestyle="-", label="_nolegend_"))

    for m in range(M_target):
        axes[0].add_patch(
            Circle(target_state_ckf[m, :2], radius_projected_ckf[m], edgecolor="blue", fill=False, lw=1,
                   linestyle="--", label="_nolegend_"))

    for n in range(N_radar):
        axes[0].add_patch(
            Circle(radar_state[n, :2], R2R, edgecolor="red", fill=False, lw=1,
                   linestyle="--", label="_nolegend_"))

    if radar_states_MPPI is not None:
        N_traj, _, horizon, _ = radar_states_MPPI.shape
        horizon = horizon - 1

        mppi_colors = (cost_MPPI - cost_MPPI.min()) / (cost_MPPI.ptp())
        mppi_color_idx = np.argsort(mppi_colors)[::-1]
        segs = radar_states_MPPI[mppi_color_idx].reshape(N_traj * N_radar, horizon+1, -1, order='F')
        segs = LineCollection(segs[:, :, :2], colors=plt.cm.jet(
            np.tile(mppi_colors[mppi_color_idx], (N_radar, 1)).T.reshape(-1, order='F')), alpha=0.5)

        if step == 0:
            cost_MPPI = np.ones(cost_MPPI.shape)


        # axes_mppi_debug.plot(radar_states_MPPI[:, n, :, 0].T, radar_states_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
        axes[0].add_collection(segs)

        # axes[0].plot(radar_state[..., 0], radar_state[..., 1], 'r*', label="Sensor Position")
        # axes[0].plot(radar_states.squeeze()[:, 1:, 0].T, radar_states.squeeze()[:, 1:, 1].T, 'r-', label="_nolegend_")
    # axes[0].plot([], [], "r.-", label="Sensor Planned Path")
    axes[0].set_title(f"time step={step}")
    # axes[0].legend(bbox_to_anchor=(0.5, 1.45),loc="upper center")
    axes[0].legend()#bbox_to_anchor=(0.7, 1.45), loc="upper center")
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')

    axes[0].axis('equal')
    axes[0].grid()

    qx, qy, logdet_grid = FIM_Visualization(ps=radar_state[:, :dm // 2], qs=target_state_true, C=C,
                                            N=1000)

    axes[1].contourf(qx, qy, logdet_grid, levels=20)
    axes[1].scatter(radar_state[:, 0], radar_state[:, 1], s=50, marker="x", color="r")
    #
    axes[1].scatter(target_state_true[..., 0].ravel(), target_state_true[..., 1].ravel(), s=50, marker="o", color="g")
    axes[1].set_title("Instant Time Objective Function Map")
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    axes[1].axis('equal')
    axes[1].grid()

    # axes[2].plot(jnp.array(FIMs), 'ko')
    # axes[2].set_ylabel("LogDet FIM (Higher is Better)")
    # axes[2].set_title(f"Avg MPPI FIM={np.round(FIMs[-1])}")

    # fig.suptitle(f"Iteration {step}")
    fig.tight_layout()
    fig.savefig(file_savepth)

    axes[0].cla()
    axes[1].cla()
    # axes[2].cla()

    return file_savepth



def plot_wireframe_sphere(center, radius,ax,color="r"):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Create data for a wireframe sphere
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the wireframe sphere
    ax.plot_wireframe(x, y, z, color=color,alpha=0.05,label="_nolegend_")

    # # Plot the center point
    # ax.scatter(center[0], center[1], center[2], color=color, s=100)

def visualize_tracking3D(target_state_true,target_state_ckf,target_states_true,
                       cost_MPPI,
                       radar_state,radar_states_MPPI,
                       R2T,R2R,
                       fig,ax,step,tmp_photo_dir,filename="visualize"):

    os.makedirs(tmp_photo_dir,exist_ok=True)
    file_savepth = os.path.join(tmp_photo_dir,filename+f"_{step}.png")

    # Main figure

    M_target,dm = target_state_true.shape
    N_radar,dn = radar_state.shape

    linecolltraj = np.swapaxes(target_states_true[:, :, :3], 1, 0)
    # linecolltraj = np.concatenate((linecolltraj,np.zeros((linecolltraj.shape[0],linecolltraj.shape[1],1))),axis=-1)
    target_traj_segs =     Line3DCollection( linecolltraj, colors="g", alpha=0.5)

    # axes[0].plot(radar_state_history[0, :, 0], radar_state_history[0, :, 1], 'md', label="Sensor Init")
    ax.scatter(target_state_true[:, 0].ravel(), target_state_true[:, 1].ravel(), target_state_true[:, 2].ravel(),color='g', marker="o",label="Target Position")
    ax.add_collection3d(target_traj_segs)
    ax.scatter(radar_state[:, 0].ravel(), radar_state[:, 1].ravel(),radar_state[:, 2].ravel(), color='r', marker="*",label="Radar Position")
    ax.scatter(target_state_ckf[:, 0].ravel(), target_state_ckf[:, 1].ravel(),target_state_ckf[:, 2].ravel(), color='b',marker="X", label="CKF Predict Position")

    # for m in range(M_target):
    #     plot_wireframe_sphere(target_state_true[m, :3], R2T, ax,color="g")

    for m in range(M_target):
        plot_wireframe_sphere(target_state_ckf[m, :3], R2T,ax,color="b")


    for n in range(N_radar):
        plot_wireframe_sphere(radar_state[n, :3], R2R, ax, color="r")

    if radar_states_MPPI is not None:
        N_traj, _, horizon, _ = radar_states_MPPI.shape
        horizon = horizon - 1

        mppi_colors = (cost_MPPI - cost_MPPI.min()) / (cost_MPPI.ptp())
        mppi_color_idx = np.argsort(mppi_colors)[::-1]
        segs = radar_states_MPPI[mppi_color_idx].reshape(N_traj * N_radar, horizon+1, -1, order='F')
        segs = Line3DCollection(segs[:, :, :3], colors=plt.cm.jet(
            np.tile(mppi_colors[mppi_color_idx], (N_radar, 1)).T.reshape(-1, order='F')), alpha=0.5)

        if step == 0:
            cost_MPPI = np.ones(cost_MPPI.shape)


        ax.add_collection3d(segs)


    ax.set_title(f"k={step}")

    ax.legend(bbox_to_anchor=(0.7, 1.45), loc="upper center")

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    ax.view_init(elev=10.)

    ax.set_aspect('equal')
    ax.grid()


    fig.tight_layout()

    fig.savefig(file_savepth)

    ax.cla()

    return file_savepth


def place_sensors(xlim,ylim,N):
    N = jnp.sqrt(N).astype(int)
    xs = jnp.linspace(xlim[0],xlim[1],N)
    ys = jnp.linspace(ylim[0],ylim[1],N)
    X,Y = jnp.meshgrid(xs,ys)
    return jnp.column_stack((X.ravel(),Y.ravel()))

def place_sensors_restricted(key,target_state,R2R,R2T,min_grid,max_grid,N_radar):
    valid_radar_positions = False
    zeros = jnp.zeros((N_radar,1))
    while not valid_radar_positions:
        ps = jax.random.uniform(key, shape=(N_radar, 2), minval=min_grid, maxval=max_grid)
        ps = jnp.concatenate((ps,zeros),axis=-1)

        R2R_distance = jnp.linalg.norm(ps[jnp.newaxis, :, :] - ps[:, jnp.newaxis, :], axis=-1)
        r2r_bool = jnp.all(R2R_distance[jnp.triu_indices_from(R2R_distance,k=1)] >= R2R)

        R2T_distance = jnp.linalg.norm(target_state[jnp.newaxis, :, :3] - ps[:, jnp.newaxis, :], axis=-1)
        r2t_bool = jnp.all(R2T_distance >= R2T)
        key, subkey = jax.random.split(key)

        if r2r_bool and r2t_bool:
            valid_radar_positions=True

    return ps,key