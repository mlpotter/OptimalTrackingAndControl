import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from src_range_publish.FIM_new.FIM_RADAR import FIM_Visualization
import numpy as np
import os


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

def visualize_tracking(target_state_true,target_state_ckf,cost_MPPI,
                       radar_state,radar_states_MPPI,
                       FIMs,
                       R2T,R2R,C,
                       fig,axes,step,tmp_photo_dir,filename="visualize"):

    os.makedirs(tmp_photo_dir,exist_ok=True)
    file_savepth = os.path.join(tmp_photo_dir,filename+f"_{step}.png")

    # Main figure

    M_target,dm = target_state_true.shape
    N_target,dn = radar_state.shape



    # axes[0].plot(radar_state_history[0, :, 0], radar_state_history[0, :, 1], 'md', label="Sensor Init")
    axes[0].plot(target_state_true[:, 0].ravel(), target_state_true[:, 1].ravel(), 'go', label="Target Position")
    axes[0].plot(radar_state[:, 0].ravel(), radar_state[:, 1].ravel(), 'r*', label="Radar")
    axes[0].plot(target_state_ckf[:, 0].ravel(), target_state_ckf[:, 1].ravel(), 'bX', label="CKF Predict")

    for m in range(M_target):
        axes[0].add_patch(
            Circle(target_state_true[m, :2], R2T[m], edgecolor="green", fill=False, lw=1,
                   linestyle="--", label="_nolegend_"))

    for n in range(N_target):
        axes[0].add_patch(
            Circle(radar_state[n, :2], R2R, edgecolor="red", fill=False, lw=1,
                   linestyle="--", label="_nolegend_"))

    if radar_states_MPPI is not None:
        N_traj, _, horizon, _ = radar_states_MPPI.shape
        horizon = horizon - 1

        mppi_colors = (cost_MPPI - cost_MPPI.min()) / (cost_MPPI.ptp())
        mppi_color_idx = np.argsort(mppi_colors)[::-1]
        segs = radar_states_MPPI[mppi_color_idx].reshape(N_traj * N_target, horizon+1, -1, order='F')
        segs = LineCollection(segs[:, :, :2], colors=plt.cm.jet(
            np.tile(mppi_colors[mppi_color_idx], (N_target, 1)).T.reshape(-1, order='F')), alpha=0.5)

        if step == 0:
            cost_MPPI = np.ones(cost_MPPI.shape)


        # axes_mppi_debug.plot(radar_states_MPPI[:, n, :, 0].T, radar_states_MPPI[:, n, :, 1].T, 'b-',label="_nolegend_")
        axes[0].add_collection(segs)

        # axes[0].plot(radar_state[..., 0], radar_state[..., 1], 'r*', label="Sensor Position")
        # axes[0].plot(radar_states.squeeze()[:, 1:, 0].T, radar_states.squeeze()[:, 1:, 1].T, 'r-', label="_nolegend_")
    # axes[0].plot([], [], "r.-", label="Sensor Planned Path")
    axes[0].set_title(f"k={step}")
    axes[0].set_title(f"k={step}")
    # axes[0].legend(bbox_to_anchor=(0.5, 1.45),loc="upper center")
    axes[0].legend(bbox_to_anchor=(0.7, 1.45), loc="upper center")
    axes[0].axis('equal')
    axes[0].grid()

    qx, qy, logdet_grid = FIM_Visualization(ps=radar_state[:, :dm // 2], qs=target_state_true, C=C,
                                            N=1000)

    axes[1].contourf(qx, qy, logdet_grid, levels=20)
    axes[1].scatter(radar_state[:, 0], radar_state[:, 1], s=50, marker="x", color="r")
    #
    axes[1].scatter(target_state_true[..., 0].ravel(), target_state_true[..., 1].ravel(), s=50, marker="o", color="g")
    axes[1].set_title("Instant Time Objective Function Map")
    axes[1].axis('equal')
    axes[1].grid()

    axes[2].plot(jnp.array(FIMs), 'ko')
    axes[2].set_ylabel("LogDet FIM (Higher is Better)")
    axes[2].set_title(f"Avg MPPI FIM={np.round(FIMs[-1])}")

    fig.suptitle(f"Iteration {step}")
    fig.tight_layout()
    fig.savefig(file_savepth)

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()

    return file_savepth

def place_sensors(xlim,ylim,N):
    N = jnp.sqrt(N).astype(int)
    xs = jnp.linspace(xlim[0],xlim[1],N)
    ys = jnp.linspace(ylim[0],ylim[1],N)
    X,Y = jnp.meshgrid(xs,ys)
    return jnp.column_stack((X.ravel(),Y.ravel()))