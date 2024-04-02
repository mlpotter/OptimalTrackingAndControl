import jax
import jax.numpy as jnp
from jax import vmap,jit
from src_range_final.tracking.Measurement_Models import RangeVelocityMeasure
@jit
def ckf_predict(X_filter, P_filter, A, Q):
    M,d = X_filter.shape
    X_filter = X_filter.reshape(-1,1)
    X_predict = A @ X_filter
    P_predict = A @ P_filter @ (A.T) + Q

    return X_predict.reshape(M,d), P_predict


@jit
def ckf_filter(measurement_actual,radar_positions,X_predict, P_predict, W):
    M,dm = X_predict.shape
    N = measurement_actual.shape[0]
    Nx = int(M*dm)

    X_predict = X_predict.reshape(-1,1)

    # U,D,_ = jnp.linalg.svd(P_predict,hermitian=False)
    # P_fact = U @ jnp.diag(jnp.sqrt(D))

    # P_fact = jax.scipy.linalg.sqrtm(P_predict)
    # P_fact = jnp.linalg.cholesky(P_predict)

    # unit_vectors = jnp.hstack((jnp.eye(P_fact.shape[0]),-jnp.eye(P_fact.shape[0]))) * jnp.sqrt(Nx)

    sigma_points = generate_sigma_points(X_predict,P_predict,Nx)

    mu = jnp.zeros((N,1))
    y_sigma_points = []
    for i in range(2*Nx):
        measurement_expected = RangeVelocityMeasure(qs=sigma_points[:,[i]].reshape(M,dm),ps=radar_positions).reshape(-1,1)
        mu += measurement_expected * 1/(2*Nx)
        y_sigma_points.append(measurement_expected)

    y_sigma_points = jnp.hstack(y_sigma_points)

    S = (y_sigma_points-mu)@(y_sigma_points-mu).T * (1/(2*Nx)) + W
    C = (sigma_points- X_predict)@(y_sigma_points-mu).T * (1/(2*Nx))

    K = C@jnp.linalg.solve(S,jnp.eye(S.shape[0]))

    X_filter = X_predict + K @ (measurement_actual-mu)

    P_filter = P_predict - K @ S @ K.T

    return X_filter.reshape(M,dm), P_filter

def generate_sigma_points(X_predict,P_predict,nx):
    X_predict = X_predict.reshape(-1,1)

    # U,D,_ = jnp.linalg.svd(P_predict,hermitian=False)
    # P_fact = U @ jnp.diag(jnp.sqrt(D))

    P_fact = jax.scipy.linalg.sqrtm(P_predict)
    # P_fact = jnp.linalg.cholesky(P_predict)

    unit_vectors = jnp.hstack((jnp.eye(P_fact.shape[0]),-jnp.eye(P_fact.shape[0]))) * jnp.sqrt(nx)

    sigma_points = X_predict + P_fact @ unit_vectors

    return sigma_points