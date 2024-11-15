import tracking.cubatureKalmanFilter as cubatureKalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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
    noise = np.random.multivariate_normal(np.zeros(dm*M_target,),Q,size=(N,)).T
    for n in range(N):
        target_state = A @ target_state + noise[:,[n]]

        x[:, n] =target_state.ravel()

    return x



def generate_data_measurement(target_states, radar_position,C,M_target,dm,N_radar):
    # range and bearing measurement
    np.random.seed(10)

    target_states = target_states.T.reshape(-1,M_target,dm)

    target_positions = target_states[:,:,:3]

    ranges = 2*np.sqrt(np.sum((radar_position[np.newaxis,:,np.newaxis,:] - target_positions[:,np.newaxis,:,:])**2,axis=-1))

    ranges = ranges.reshape(-1,M_target*N_radar).T

    z = ranges + np.random.randn(*ranges.shape) * np.sqrt(C*ranges ** 4)

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

def generate_measurement_noisy(target_state, radar_position,C,M_target,dm,N_radar):
    # range and bearing measurement
    target_state = target_state.reshape(M_target,dm)

    target_position = target_state[:,:3]

    ranges = 2*np.sqrt(np.sum((radar_position[np.newaxis,:,:] - target_position[:,np.newaxis,:])**2,axis=-1))

    ranges = ranges.ravel()

    z = ranges + np.random.randn(*ranges.shape) * np.sqrt(C*(ranges/2) ** 4)

    return z


# Main
if __name__ == '__main__':
    np.random.seed(10)
    N  = 1000

    # ==================== RADAR PARAMETERS ========================= #
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
    K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * np.pi) ** 3
    Pr = K / (R ** 4)

    SNR = -0


    # ================== TARGET and RADAR STATES ======================== #
    N_radar = 8

    # radar_position = place_sensors([-400, 400], [-400, 400], N_radar)
    radar_position = np.random.uniform(size=(N_radar, 2), low=-300, high=300)
    radar_position = np.column_stack((radar_position, np.array([0] * N_radar).reshape(-1, 1)))

    z_elevation = 100

    # target_state = np.array([[0.0, -0.0, z_elevation, 25., 20, 0],  # ,#,
    #                          [-50.4, 30.32, z_elevation+10, -20, -10, 0],
    #                          [20.4, 20.32, z_elevation-10, 25, 10, 0]])  # , #,


    target_state = np.array([[0.0, -0.0,z_elevation+10, 25., 20,0], #,#,
                    [-100.4,-30.32,z_elevation-15,20,-10,0], #,
                    [30,30,z_elevation+20,-10,-10,0]])#,

    M_target, dm = target_state.shape;

    target_state = target_state.ravel()


    # ================== NOISE PARAMETERS ============================== #
    sigmaQ = np.sqrt(10 ** (3));

    sigmaW = np.sqrt(M_target * Pr / (10 ** (SNR / 10)))
    C = c ** 2 * sigmaW ** 2 / (fc ** 2 * 8 * np.pi ** 2) * 1 / K

    # ================== DYNAMIC MODEL PARAMETERS ====================== #
    # dt = 0.1
    #
    # A_single = np.array([[1., 0, 0, dt, 0, 0],
    #                       [0, 1., 0, 0, dt, 0],
    #                       [0, 0, 1, 0, 0, dt],
    #                       [0, 0, 0, 1, 0, 0],
    #                       [0, 0, 0, 0, 1., 0],
    #                       [0, 0, 0, 0, 0, 1]])
    dt = 0.025
    Q_single = np.array([
        [(dt ** 4) / 4, 0, 0, (dt ** 3) / 2, 0, 0],
        [0, (dt ** 4) / 4, 0, 0, (dt ** 3) / 2, 0],
        [0, 0, (dt ** 4) / 4, 0, 0, (dt ** 3) / 2],
        [(dt ** 3) / 2, 0, 0, (dt ** 2), 0, 0],
        [0, (dt ** 3) / 2, 0, 0, (dt ** 2), 0],
        [0, 0, (dt ** 3) / 2, 0, 0, (dt ** 2)]
    ]) * sigmaQ ** 2

    # A = np.kron(np.eye(M_target), A_single);
    Q = np.kron(np.eye(M_target), Q_single);
    G = np.eye(N_radar)


    true_target_state = generate_data_state(target_state,N, M_target, dm,dt,Q)

    # plt.figure()
    # plt.plot(true_target_state.T.reshape(-1,M_target,dm)[:,:,0],true_target_state.T.reshape(-1,M_target,dm)[:,:,1])
    # plt.show()

    measurement = generate_data_measurement(true_target_state, radar_position,C,M_target,dm,N_radar)

    ckf = cubatureKalmanFilter.CubatureKalmanFilter(dim_x=M_target*dm, dim_z=M_target*N_radar, dt=dt,
                                                     hx=measurement_model, fx=transition_model)


    np.savetxt("measurement.csv", measurement, delimiter=",", fmt='%.22e')
    # initialze Cubature filter
    x_est_ckf = np.zeros((dm*M_target, N))

    ckf.x = np.array(target_state) + np.random.randn(M_target*dm) * 25
    ckf.x = ckf.x.reshape(-1,1)

    target_state_init = deepcopy(ckf.x)
    print("True Target State Init: ",target_state)
    print("CKF Target State Init: ",target_state_init)

    ckf.P = np.eye(M_target*dm) * 25
    ckf.Q = Q #+ np.eye(M_target*dm)*1e-4
    # ckf.R = np.diag(np.ones(dm*M_target))

    np.savetxt("x0.csv", ckf.x, delimiter=",", fmt='%.22e')
    # filtering
    for n in range(N):

        # CKF
        ckf.predict(fx_args=(M_target,))
        # measurement_next_expected = measurement_model(true_target_state[:,n],radar_position)
        measurement_next_expected = measurement_model(ckf.x_prior.ravel(),radar_position,M_target,dm,N_radar)

        ckf.R = np.diag(C*(measurement_next_expected/2) ** 4)

        # ckf.predict_propogate(ckf.x, ckf.P, 10, dt=dt*5, fx_args=(M_target,))
        range_actual = measurement_model(true_target_state[:,n], radar_position[:,:3], M_target, dm,N_radar)

        measurement = range_actual + np.random.randn()*(C*(range_actual.ravel()/2) ** 4)
        # ckf.update(np.reshape(measurement[:, n],(-1,1)), hx_args=(radar_position,M_target,dm,N_radar))
        ckf.update(np.reshape(measurement,(-1,1)), hx_args=(radar_position,M_target,dm,N_radar))

        x_est_ckf[:, n] = ckf.x.reshape(M_target*dm, )

    print("Final Target State: ",ckf.x.ravel())
    np.savetxt("x_est_ckf.csv", x_est_ckf, delimiter=",", fmt='%.22e')
    # plot
    plt.figure()
    plt.plot(x_est_ckf.T.reshape(-1,M_target,dm)[:,:,0], x_est_ckf.T.reshape(-1,M_target,dm)[:,:,1], 'b-o', label='CKF estimate',alpha=0.5)
    plt.plot(target_state_init.reshape(M_target,dm)[:,0], target_state_init.reshape(M_target,dm)[:,1], 'gX', label='CKF Initial Guess',alpha=0.5)
    plt.plot(true_target_state.T.reshape(-1,M_target,dm)[:,:,0], true_target_state.T.reshape(-1,M_target,dm)[:,:,1], 'g-', label='true')
    plt.plot(radar_position[:,0], radar_position[:,1], 'md', label='Radar Position')

    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x_est_ckf.T.reshape(-1,M_target,dm)[:,:,2], 'b-o', label='CKF estimate Elevation',alpha=0.5)
    plt.legend()
    plt.show()