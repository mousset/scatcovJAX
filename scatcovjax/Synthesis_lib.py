import time
import numpy as np
import optax
import jax
import jax.numpy as jnp
from jax import jit, grad
import jaxopt

"""
Few functions that can be used to run a synthesis.
"""

def chi2(model, data):
    return jnp.sum(jnp.abs(data-model)**2)


###### Minimizers
def fit_jaxopt(params, loss_func, method='GradientDescent', niter: int = 10, loss_history: list = None):
    print('Starting loss:', loss_func(params))

    if method == 'LBFGS':
        optimizer = jaxopt.LBFGS(fun=loss_func, jit=True)
    elif method == 'GradientDescent':
        optimizer = jaxopt.GradientDescent(fun=loss_func, jit=True)

    if loss_history is None:
        loss_history = []
        loss_history.append(loss_func(params))

    opt_state = optimizer.init_state(params)
    for i in range(niter):
        start = time.time()
        params, opt_state = optimizer.update(params, opt_state)
        end = time.time()
        if i % 10 == 0:
            loss_value = loss_func(params)
            loss_history.append(loss_value)
            print(f'Iter {i}, Loss: {loss_value:.10f}, Time = {end - start:.5f} s/iter')

    return params, loss_history


def fit_jaxopt_Scipy(params, loss_func, method='L-BFGS-B', niter: int = 10, loss_history: list = None):
    if loss_history is None:
        loss_history = []
        loss_history.append(loss_func(params))

    optimizer = jaxopt.ScipyMinimize(fun=loss_func, method=method, jit=True, maxiter=1)

    for i in range(niter):
        start = time.time()
        params, opt_state = optimizer.run(params)
        end = time.time()
        if i % 10 == 0:
            loss_history.append(opt_state.fun_val)
            print(
                f'Iter {i}, Success: {opt_state.success}, Loss = {opt_state.fun_val}, Time = {end - start:.5f} s/iter')

    return params, loss_history


def fit_optax(params: optax.Params, optimizer: optax.GradientTransformation, loss_func,
              niter: int = 10, loss_history: list = None) -> optax.Params:
    ### Gradient of the loss function
    grad_func = jit(grad(loss_func))

    if loss_history is None:
        loss_history = []
    opt_state = optimizer.init(params)
    for i in range(niter):
        start = time.time()
        grads = jnp.conj(grad_func(params))  # Take the conjugate of the gradient
        #grads = grad_func(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        end = time.time()
        if i % 10 == 0:
            loss_value = loss_func(params)
            loss_history.append(loss_value)
            print(f'Iter {i}, Loss: {loss_value:.10f}, Time = {end - start:.5f} s/iter')

    return params, loss_history


def fit_brutal(params, loss_func, momentum: float = 2., niter: int = 10, loss_history: list = None):
    ### Gradient of the loss function
    grad_loss_func = jit(grad(loss_func))

    if loss_history is None:
        loss_history = []
    for i in range(niter):
        start = time.time()
        params -= momentum * np.conj(grad_loss_func(params))
        #params -= momentum * grad_loss_func(params)
        if i % 10 == 0:
            end = time.time()
            loss_value = loss_func(params)
            loss_history.append(loss_value)
            print(f"Iter {i}: Loss = {loss_value:.5f}, Momentum = {momentum}, Time = {end - start:.2f} s/iter")

    return params, loss_history

