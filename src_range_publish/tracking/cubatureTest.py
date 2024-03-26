import cubatureKalmanFilter
import squareRootCubatureKalmanFilter
import numpy as np
import matplotlib.pyplot as plt


# Generate data
def generate_data_state(N, dt):
    # 2D constant velocity model
    t = np.arange(0, N*dt, dt)
    p0 = np.array([-40, -5]) # initial position
    v0 = np.array([5, 1]) # initial velocity
    x = np.zeros((4, N)) # state

    for n in range(N):
        if n == 0:
            p = p0 + v0*dt
        else:
            p = p + v0*dt
        v = v0
        x[:, n] = np.concatenate((p, v), axis=0) # state

    return x


def generate_data_measurement(x, ref_point, noise):
    # range and bearing measurement
    np.random.seed(10)
    rang = np.sqrt((x[0, :] - ref_point[0])**2 + (x[1, :] - ref_point[1])**2) + noise[0]*np.random.randn(1, x.shape[1])
    bearing = np.arctan2(x[1, :] - ref_point[1], x[0, :] - ref_point[0]) + noise[1]*np.random.randn(1, x.shape[1])
    z = np.concatenate((rang, bearing), axis=0)
    return z


def transition_model(x, dt):
    # 2D constant velocity model
    # x = [px, py, vx, vy]
    x[0] = x[0] + x[2]*dt
    x[1] = x[1] + x[3]*dt
    return x


def measurement_model(x, ref_point):
    # x = [px, py, vx, vy]
    # ref_point = [px, py]
    rang = np.sqrt((x[0] - ref_point[0])**2 + (x[1] - ref_point[1])**2)
    bearing = np.arctan2(x[1] - ref_point[1], x[0] - ref_point[0])
    z = np.array([rang, bearing])
    return z


# Main
if __name__ == '__main__':
    np.random.seed(10)
    N  = 100
    dt = 0.1
    true_state = generate_data_state(N, dt)
    ref_point = np.array([-10, 5])
    sigma = np.array([0.05, 0.01])
    measurement = generate_data_measurement(true_state, ref_point, sigma)
    ckf = cubatureKalmanFilter.CubatureKalmanFilter(4, 2, dt,
                                                     hx=measurement_model, fx=transition_model)
    sckf = squareRootCubatureKalmanFilter.SquareRootCubatureKalmanFilter(4, 2, dt,
                                                     hx=measurement_model, fx=transition_model)
    np.savetxt("measurement.csv", measurement, delimiter=",", fmt='%.22e')
    # initialze Cubature filter
    x_est_ckf = np.zeros((4, N))
    x_est_sckf = np.zeros((4, N))

    ckf.x = np.array([-40, -5, 5, 1]) + np.random.randn(4) * 1
    ckf.P = np.eye(4) * 1
    ckf.Q = np.eye(4) * 0.0001
    ckf.R = np.diag(sigma) ** 2
    sckf.x = np.array([-40, -5, 5, 1]) + np.random.randn(4) * 1
    sckf.P = np.eye(4) * 1
    sckf.Q = np.eye(4) * 0.0001
    sckf.R = np.diag(sigma) ** 2
    np.savetxt("x0.csv", sckf.x, delimiter=",", fmt='%.22e')
    # filtering
    for n in range(N):
        # CKF
        ckf.predict()
        ckf.update(np.reshape(measurement[:, n], (-1, 1)), hx_args=(ref_point))
        x_est_ckf[:, n] = ckf.x.reshape(4, )

        # SCKF
        sckf.predict()
        sckf.update(np.reshape(measurement[:, n], (-1, 1)), hx_args=(ref_point))
        x_est_sckf[:, n] = sckf.x.reshape(4, )
    np.savetxt("x_est_sckf.csv", x_est_sckf, delimiter=",", fmt='%.22e')
    # plot
    plt.figure()
    plt.plot(true_state[0, :], true_state[1, :], 'g-', label='true')
    plt.plot(x_est_ckf[0, :], x_est_ckf[1, :], 'b-*', label='CKF estimate')
    plt.plot(x_est_sckf[0, :], x_est_sckf[1, :], 'r-*', label='SCKF estimate')
    plt.legend()
    plt.show()





