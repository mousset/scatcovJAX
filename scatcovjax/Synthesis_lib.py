import time
import numpy as np
import optax
import jax
import jax.numpy as jnp
from jax import jit, grad

"""
Few functions that can be used to run a synthesis.
"""


def chi2(model, data):
    return jnp.sum(jnp.abs(data-model)**2)


def fit_brutal(params, loss_func, momentum: float = 2., niter: int = 10):
    ### Gradient of the loss function
    grad_loss_func = jit(grad(loss_func))
    # grad_loss_func = grad(loss_func)

    loss_history = []
    for i in range(niter):
        start = time.time()
        params -= momentum * np.conj(grad_loss_func(params))
        if i % 1 == 0:
            end = time.time()
            loss_value = loss_func(params)
            loss_history.append(loss_value)
            print(f"Iter {i}: Loss = {loss_value:.5f}, Momentum = {momentum}, Time = {end - start:.2f} s/iter")

    return params, loss_history


def fit_optax(params: optax.Params, optimizer: optax.GradientTransformation, loss_func, niter : int =10) -> optax.Params:
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss_func)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    loss_history = []
    for i in range(niter):
        start = time.time()
        params, opt_state, loss_value = step(params, opt_state)
        if i % 1 == 0:
            loss_history.append(loss_value)
            end = time.time()
            print(f'Iter {i}, Loss: {loss_value:.5f}, Time = {end - start:.5f} s/iter')

    return params, loss_history
