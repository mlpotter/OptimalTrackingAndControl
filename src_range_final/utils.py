import jax.numpy as jnp
def NoiseParams(Pr,SCNR,CNR=-10):
    A = jnp.sqrt(Pr);
    zeta = jnp.sqrt(A**2 / ((1+10**(CNR/10)) * 10**(SCNR/10)) * 0.5)
    Ma = jnp.sqrt((A**2 * 10**(CNR/10)) / ((1+10**(CNR/10)) * 10**(SCNR/10)))

    s = jnp.sqrt(0.5*Ma**2 + zeta**2);

    return A,Ma,zeta,s


def SNR2PARAM(SNR, Pr):
    K = 10 ** (SNR / 10)
    sigma = np.sqrt(Pr / (2 * K + 2))
    s = np.sqrt(K * Pr / (K + 1))
    return sigma, s


def SCIPY2RICE(sigma, s):
    b = s / sigma
    scale = sigma
    loc = np.zeros_like(b)
    return b, scale, loc


def rice_power(Pr, K, Pr_avg):
    pdf = (K + 1) / Pr_avg * jnp.exp(-K - (K + 1) * Pr / Pr_avg) * jnp.i0(
        2 * jnp.sqrt(K * (K + 1) / Pr_avg) * jnp.sqrt(Pr))
    return pdf


def place_sensors(xlim,ylim,N):
    N = jnp.sqrt(N).astype(int)
    xs = jnp.linspace(xlim[0],xlim[1],N)
    ys = jnp.linspace(ylim[0],ylim[1],N)
    X,Y = jnp.meshgrid(xs,ys)
    return jnp.column_stack((X.ravel(),Y.ravel()))