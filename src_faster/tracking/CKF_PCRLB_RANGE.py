import src_range_publish.tracking.cubatureKalmanFilter as cubatureKalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from src_range_publish.FIM_new.FIM_RADAR import Single_FIM_Radar,Single_JU_FIM_Radar,FIM_Visualization,JU_RANGE_FIM
from src_range_publish.FIM_new.FIM_RADAR import JU_RANGE_FIM
from jax.tree_util import Partial as partial

from tqdm import tqdm

from math import isclose

def place_sensors(xlim,ylim,N):
    N = np.sqrt(N).astype(int)
    xs = np.linspace(xlim[0],xlim[1],N)
    ys = np.linspace(ylim[0],ylim[1],N)
    X,Y = np.meshgrid(xs,ys)
    return np.column_stack((X.ravel(),Y.ravel()))

# Generate data
def generate_data_state(target_state,N, M_target, dm, dt,Q):

    # 2D constant velocity model
    A_single = np.array([[1., 0, 0, dt, 0, 0],
                         [0, 1., 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1., 0],
                         [0, 0, 0, 0, 0, 1]])

    A = np.kron(np.eye(M_target), A_single);

    target_state = target_state.reshape(-1,1)

    x = np.zeros((dm*M_target, N)) # state

    if isclose(np.sum(Q),0):
        noises = np.zeros((N,Q.shape[0]))
        print("here?")
    else:
        noises = np.random.multivariate_normal(np.zeros(A.shape[0]), Q, size=(N,)).T

    for n in range(N):
        target_state = A @ target_state

        x[:, n] = target_state.ravel()  + noises[:,n]

    return x



def generate_data_measurement(target_states, radar_position,M_target,dm,N_radar,sigmaR):
    # range and bearing measurement
    np.random.seed(10)

    target_states = target_states.T.reshape(-1,M_target,dm)

    target_positions = target_states[:,:,:3]

    ranges = 2*np.sqrt(np.sum((radar_position[np.newaxis,:,np.newaxis,:] - target_positions[:,np.newaxis,:,:])**2,axis=-1))

    ranges = ranges.reshape(-1,M_target*N_radar).T

    z = ranges + np.random.randn(*ranges.shape) * sigmaR

    return z


def transition_model(x, dt,M_target):
    # 2D constant velocity model
    # x = [px, py, vx, vy]

    A_single = np.array([[1., 0, 0, dt, 0, 0],
                         [0, 1., 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1., 0],
                         [0, 0, 0, 0, 0, 1]])

    A = np.kron(np.eye(M_target), A_single);

    x = A @ x.reshape(-1,1)
    x = x.ravel()

    return x


def measurement_model(target_state, radar_position,M_target,dm,N_radar):
    # x = [px, py, vx, vy]
    # ref_point = [px, py]

    target_state = target_state.T.reshape(M_target,dm)

    target_position = target_state[:,:3]

    ranges = 2*np.sqrt(np.sum((radar_position[:,np.newaxis,:] - target_position[np.newaxis,:,:])**2,axis=-1))

    z = ranges.ravel()
    return z

def generate_measurement_noisy(target_state, radar_position,M_target,dm,N_radar,sigmaR):
    # range and bearing measurement
    target_state = target_state.reshape(M_target,dm)

    target_position = target_state[:,:3]

    ranges = 2*np.sqrt(np.sum((radar_position[np.newaxis,:,:] - target_position[:,np.newaxis,:])**2,axis=-1))

    ranges = ranges.ravel()

    z = ranges + np.random.randn(*ranges.shape) * sigmaR

    return z


# Main
if __name__ == '__main__':
    np.random.seed(10)
    N  = 500

    # ================== TARGET and RADAR STATES ======================== #
    N_radar = 4

    # radar_position = place_sensors([-400, 400], [-400, 400], N_radar)
    radar_position = np.random.uniform(size=(N_radar, 2), low=-50, high=50)
    radar_position = np.column_stack((radar_position, np.array([0] * N_radar).reshape(-1, 1)))

    z_elevation = 100

    # target_state = np.array([[0.0, -0.0, z_elevation, 25., 20, 0],  # ,#,
    #                          [-50.4, 30.32, z_elevation+10, -20, -10, 0],
    #                          [20.4, 20.32, z_elevation-10, 25, 10, 0]])  # , #,


    target_state = np.array([[0.0, -0.0,z_elevation+10, 0., 0,0]])#, #,#,
                    #[-100.4,-30.32,z_elevation-15,0,0,0]]) #,
                    # [30,30,z_elevation+20,-10,-10,0]])#,

    M_target, dm = target_state.shape;

    target_state = target_state.ravel()


    # ================== NOISE PARAMETERS ============================== #
    sigmaQ = 15
    sigmaR = 1
    # ================== DYNAMIC MODEL PARAMETERS ====================== #
    # dt = 0.1
    #
    # A_single = np.array([[1., 0, 0, dt, 0, 0],
    #                       [0, 1., 0, 0, dt, 0],
    #                       [0, 0, 1, 0, 0, dt],
    #                       [0, 0, 0, 1, 0, 0],
    #                       [0, 0, 0, 0, 1., 0],
    #                       [0, 0, 0, 0, 0, 1]])
    dt = 0.1
    Q_single = np.array([
        [(dt ** 4) / 4, 0, 0, (dt ** 3) / 2, 0, 0],
        [0, (dt ** 4) / 4, 0, 0, (dt ** 3) / 2, 0],
        [0, 0, (dt ** 4) / 4, 0, 0, (dt ** 3) / 2],
        [(dt ** 3) / 2, 0, 0, (dt ** 2), 0, 0],
        [0, (dt ** 3) / 2, 0, 0, (dt ** 2), 0],
        [0, 0, (dt ** 3) / 2, 0, 0, (dt ** 2)]
    ]) * sigmaQ ** 2

    # 2D constant velocity model
    A_single = np.array([[1., 0, 0, dt, 0, 0],
                         [0, 1., 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1., 0],
                         [0, 0, 0, 0, 0, 1]])

    A = np.kron(np.eye(M_target), A_single);

    # A = np.kron(np.eye(M_target), A_single);
    Q = np.kron(np.eye(M_target), Q_single);
    Q = np.eye(Q.shape[0])*1
    R = np.eye(N_radar*M_target)*sigmaR**2

    trials = 50
    RMSE_bound = np.zeros((N,))
    RMSE = np.zeros((N,trials))



    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > -1e-8)

    for trial in tqdm(range(trials)):
        true_target_state = generate_data_state(target_state, N, M_target, dm, dt, Q)

        measurements = generate_data_measurement(true_target_state, radar_position, M_target, dm, N_radar, sigmaR)

        ckf = cubatureKalmanFilter.CubatureKalmanFilter(dim_x=M_target * dm, dim_z=M_target * N_radar, dt=dt,
                                                        hx=measurement_model, fx=transition_model)


        ckf.x = np.array(target_state) + np.random.randn(M_target * dm) * 25
        ckf.x = ckf.x.reshape(-1, 1)

        target_state_init = deepcopy(ckf.x)

        ckf.P = np.eye(M_target * dm) * 15
        ckf.Q = Q
        ckf.R = R

        # ckf.R = np.diag(np.ones(dm*M_target))

        J = deepcopy(np.linalg.inv(ckf.P))

        IM_fn = partial(JU_RANGE_FIM, A=A, Q=Q, R=R)

        x_est_ckf = np.zeros((dm * M_target, N))
        # filtering
        for n in range(N):
            # CKF
            ckf.predict(fx_args=(M_target,))
            # measurement_next_expected = measurement_model(true_target_state[:,n],radar_position)

            # ckf.predict_propogate(ckf.x, ckf.P, 10, dt=dt*5, fx_args=(M_target,))
            # range_actual = measurement_model(true_target_state[:,n], radar_position[:,:3], M_target, dm,N_radar)

            # measurement = range_actual + np.random.randn()*sigmaR
            measurement_true = measurements[:,n]

            # ckf.update(np.reshape(measurement[:, n],(-1,1)), hx_args=(radar_position,M_target,dm,N_radar))
            ckf.update(np.reshape(measurement_true,(-1,1)), hx_args=(radar_position,M_target,dm,N_radar))


            J = IM_fn(radar_state=radar_position.reshape(N_radar,3),target_state=true_target_state[:,n].reshape(M_target,dm),J=J) #[JU_FIM_D_Radar(ps=ps, q=m0[[i],:], Pt=Pt, Gt=Gt, Gr=Gr, L=L, lam=lam, rcs=rcs, A=A_single, Q=Q_single, J=Js[i],s=s) for i in range(len(Js))]

            PCRLB = np.linalg.solve(J,np.eye(J.shape[0]))

            RMSE_bound_n = np.trace(PCRLB)
            # print(RMSE_bound_n)
            RMSE_bound[n] = np.sqrt(RMSE_bound_n)

            RMSE[n,trial] = np.sum((ckf.x.ravel() - true_target_state[:,n].ravel())**2) #np.sqrt(np.trace(ckf.P))#np.sqrt(np.sum((ckf.x.ravel() - true_target_state[:,n].ravel())**2))
            x_est_ckf[:, n] = ckf.x.reshape(M_target * dm, )
            # print(f"N={n} ",np.sum((ckf.x.ravel()[4] - true_target_state[:,n].ravel()[4])**2),PCRLB[4,4])
            # print((ckf.x.ravel() - true_target_state[:,n].ravel())**2)
            # print(np.diag(PCRLB))
            # print(is_pos_def(ckf.P-PCRLB),np.linalg.eigvals(ckf.P-PCRLB).min(),np.linalg.eigvals(ckf.P-PCRLB).max())
            # print("P: ",n,np.linalg.eigvals(ckf.P).min(),np.linalg.eigvals(ckf.P).max())
            # print("PCRLB: ",n,np.linalg.eigvals(PCRLB).min(),np.linalg.eigvals(PCRLB).max())

    RMSE = np.sqrt(RMSE.mean(axis=1)).ravel()
    RMSE_bound = RMSE_bound.ravel()
    print("True Target State: ",true_target_state[:,n])
    print("Final Target State: ",ckf.x.ravel())

    plt.figure()
    plt.plot(x_est_ckf.T.reshape(-1,M_target,dm)[:,:,0], x_est_ckf.T.reshape(-1,M_target,dm)[:,:,1], 'b-o', label='CKF estimate',alpha=0.5)
    plt.plot(target_state_init.reshape(M_target,dm)[:,0], target_state_init.reshape(M_target,dm)[:,1], 'gX', label='CKF Initial Guess',alpha=0.5)
    plt.plot(true_target_state.T.reshape(-1,M_target,dm)[:,:,0], true_target_state.T.reshape(-1,M_target,dm)[:,:,1], 'go-', label='true')
    plt.plot(radar_position[:,0], radar_position[:,1], 'md', label='Radar Position')

    plt.legend()
    plt.show()


    print("Final Target State: ",ckf.x.ravel())

    fig,axes = plt.subplots(1,3,figsize=(10,5))
    axes[0].plot(RMSE_bound,label="PCRLB RMSE Bound")
    # axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].plot(RMSE_bound,label="PCRLB RMSE Bound")
    axes[1].plot(RMSE,label="CKF RMSE")
    # axes[1].set_yscale("log")
    axes[1].legend()
    axes[2].plot(RMSE-RMSE_bound,label="Difference RMSE")
    # axes[2].set_yscale("log")
    axes[2].legend()
    fig.show()