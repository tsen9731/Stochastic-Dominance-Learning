import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

def sd_1st_cdf(x, y, rel_tau=0.3, get_utility=False):
    """First-order stochastic dominance loss. Approximate 1(x>0) by 

    Args:
        x, y: Scalar array containing samples from two distributions, which we want to maximize.
        rel_tau: Softmax temperature control
        get_utility: Return array u(x) instead of a scalar loss

    Returns:
        Loss value to minimize, or the utility function u(x)
    """    
    nX, nY = len(x), len(y)
    is_y = jnp.concatenate([jnp.zeros_like(x, dtype=float), jnp.ones_like(y, dtype=float)])
    values = jnp.concatenate([x, y])
    idx_sort = jnp.argsort(values)
    sorted_is_y = is_y[idx_sort]
    sorted_values = values[idx_sort]

    F1x = jnp.cumsum(1-sorted_is_y)/nX
    F1y = jnp.cumsum(sorted_is_y)/nY
    eps = jnp.finfo(x.dtype).eps
    
    eta_values = stop_gradient(sorted_values + eps)
    # mu = stop_gradient(F1x > F1y).astype(x.dtype)
    tau = (jnp.max(F1x - F1y) - jnp.min(F1x - F1y))*rel_tau
    mu = jnp.exp(((F1x - F1y) - jnp.max(F1x - F1y))/tau)
    mu = mu/(jnp.sum(mu)+eps)
    mu = stop_gradient(mu).astype(x.dtype)

    eta = jnp.expand_dims(eta_values, axis=1)

    # Previous code (Dai 2023) suggests relu
    if get_utility:
        ux = jnp.sum(jax.nn.relu(eta - jnp.expand_dims(x, axis=0))*jnp.expand_dims(mu, axis=1), axis=0)
        return ux
    else:
        ex = jnp.mean(jax.nn.relu(eta - jnp.expand_dims(x, axis=0)), axis=1)
        # ex = jnp.mean(jax.nn.sigmoid(eta - jnp.expand_dims(x, axis=0)), axis=1)
        loss = jnp.sum(ex*mu)
        return loss

def sd_2nd_cdf(x, y, rel_tau=0.3, get_utility=False):
    """Second-order stochastic dominance loss.

    Args:
        x, y: Scalar array containing samples from two distributions, which we want to maximize.
        rel_tau: Softmax temperature control
        get_utility: Return array u(x) instead of a scalar loss

    Returns:
        Loss value to minimize, or the utility function u(x)
    """    
    nX, nY = len(x), len(y)
    is_y = jnp.concatenate([jnp.zeros_like(x, dtype=float), jnp.ones_like(y, dtype=float)])
    eta = jnp.concatenate([x, y])
    idx_sort = jnp.argsort(eta)
    sorted_is_y = is_y[idx_sort]
    sorted_eta = eta[idx_sort]

    F1x = jnp.cumsum(1-sorted_is_y)/nY
    F1y = jnp.cumsum(sorted_is_y)/nX
    h = sorted_eta - jnp.roll(sorted_eta,1)

    F2x_incre = h*jnp.roll(F1x,1)
    F2y_incre = h*jnp.roll(F1y,1)
    F2x_incre = F2x_incre.at[0].set(0)
    F2y_incre = F2y_incre.at[0].set(0)

    F2x = jnp.cumsum(F2x_incre)
    F2y = jnp.cumsum(F2y_incre)

    # correct gradient computation
    F2x = F2x + (1-sorted_is_y)*F1x*(stop_gradient(sorted_eta)-sorted_eta)
    F2y = stop_gradient(F2y)

    eps = jnp.finfo(x.dtype).eps

    tau = (jnp.max(F2x - F2y) - jnp.min(F2x - F2y))*rel_tau
    mu = jnp.exp(((F2x - F2y) - jnp.max(F2x - F2y))/tau)
    mu = mu/(jnp.sum(mu)+eps)
    mu = stop_gradient(mu).astype(x.dtype)

    if get_utility:
        u1 = jnp.cumsum(mu[::-1])[::-1]
        u2_incre = (jnp.roll(sorted_eta,-1) - sorted_eta)*jnp.roll(u1,-1)
        u2_incre = u2_incre.at[-1].set(0)
        u2 = jnp.cumsum(u2_incre[::-1])[::-1]
        ux = u2[jnp.argsort(idx_sort)[:nX]]
        return ux
    else:
        loss = jnp.sum((F2x - F2y)*mu)
        return loss

# Straightforward O(N^2) implementation
# def sd_2nd_cdf_(x, y, rel_tau=0.3, get_utility=False):

#     values = jnp.concatenate([x, y])
#     eps = jnp.finfo(x.dtype).eps
    
#     eta = stop_gradient(jnp.expand_dims(values, axis=1))

#     F2x = jnp.mean(jax.nn.relu(eta - jnp.expand_dims(x, axis=0)), axis=1)
#     F2y = jnp.mean(jax.nn.relu(eta - jnp.expand_dims(y, axis=0)), axis=1)

#     tau = (jnp.max(F2x - F2y) - jnp.min(F2x - F2y))*rel_tau
#     mu = jnp.exp(((F2x - F2y) - jnp.max(F2x - F2y))/tau)
#     mu = mu/(jnp.sum(mu)+eps)
#     mu = stop_gradient(mu)

#     if get_utility:
#         ux = jnp.sum(jax.nn.relu(eta - jnp.expand_dims(x, axis=0))*jnp.expand_dims(mu, axis=1), axis=0)
#         return ux
#     else:
#         loss = jnp.sum((F2x - F2y)*mu)
#         return loss

def mean_risk(x):
    mean = jnp.mean(x)
    return mean + 0.5*jnp.mean(jnp.abs(x-mean))