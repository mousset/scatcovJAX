# Specify CUDA device
from jax import jit, config

config.update("jax_enable_x64", True)

# Check we're running on GPU
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)

import os
import argparse
import time
import numpy as np
from jax import jit, grad
import jax.numpy as jnp

import scatcovjax.Sphere_lib as sphlib
from scatcovjax.Scattering_lib import scat_cov_axi, scat_cov_dir
from s2wav.filter_factory.filters import filters_directional_vectorised

import s2fft
import s2wav

"""
Script to run a synthesis. To run it do:
    $ python run_synthesis.py --L --N --epochs --save_dir

"""

########## ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument('-l', '--L', help='L max value', default=32, type=int)
parser.add_argument('-n', '--N', help='N value', default=3, type=int)
parser.add_argument('-e', '--epochs', help='Number of epochs', default=10, type=int)
parser.add_argument('-s', '--save_dir', help='Path where outputs are saved.', default='./', type=str)

args = parser.parse_args()

# If save_dir does not exist we create the directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

L = args.L
N = args.N
axi = True if N == 1 else False  # Axisym or directional
J = s2wav.utils.shapes.j_max(L)

########## DEFAULT PARAMETERS
sampling = "mw"
multiresolution = True
reality = True
J_min = 0
momentum = 2.  # For gradient descent
planet = 'venus'

###### Loss function
def chi2(model, data):
    return jnp.sum(jnp.abs(data - model) ** 2)


def func(flm):
    if axi:
        mean_new, var_new, S1_new, P00_new, C01_new, C11_new = scat_cov_axi(flm, L, N, J_min, sampling,
                                                                            None, reality, multiresolution,
                                                                            normalisation=None, filters=filters)
    else:
        mean_new, var_new, S1_new, P00_new, C01_new, C11_new = scat_cov_dir(flm, L, N, J_min, sampling,
                                                                            None, reality, multiresolution,
                                                                            normalisation=None, filters=filters)
    # Control for mean + var
    loss = chi2(mean, mean_new)
    loss += chi2(var, var_new)

    # Add S1 loss
    loss += chi2(S1, S1_new)

    # Add P00 loss
    loss += chi2(P00, P00_new)

    # Add C01 loss
    loss += chi2(C01, C01_new)

    # Add C11 loss
    loss += chi2(C11, C11_new)

    return loss


######################################################################################


if __name__ == "__main__":
    ###### Print the parameters
    print('\n============ Parameters ===============')
    print(f'{axi=} , {L=}, {N=}, {J=}, {J_min=}, {reality=}, {planet=}')

    ###### Build filters
    print('\n============ Build the filters ===============')
    filters = filters_directional_vectorised(L, N, J_min)

    ###### Target
    print('\n============ Make the target ===============')
    I, Ilm = sphlib.make_MW_planet(L, planet, normalize=True, reality=reality)
    if axi:
        mean, var, S1, P00, C01, C11 = scat_cov_axi(Ilm, L, N, J_min, sampling, None,
                                                    reality, multiresolution, filters=filters)
    else:
        mean, var, S1, P00, C01, C11 = scat_cov_dir(Ilm, L, N, J_min, sampling, None,
                                                    reality, multiresolution, filters=filters)

    ##### Initial condition
    print('\n============ Build initial conditions ===============')
    f = np.random.randn(L, 2 * L - 1).astype(np.float64)
    flm = s2fft.forward_jax(f, L, reality=reality)
    flm = flm[:, L - 1:] if reality else flm

    flm_start = jnp.copy(flm)  # Save the start point as we will iterate on flm
    loss_0 = func(flm_start)  # Compute the starting loss

    ##### Run synthesis
    print('\n============ Compute the loss gradient ===============')
    # grad_func = jit(grad(func))  # Compute the loss gradient
    grad_func = grad(func)

    print('\n============ Start the synthesis ===============')
    loss_history = [loss_0]
    for i in range(args.epochs):
        start = time.time()
        flm -= momentum * np.conj(grad_func(flm))
        if i % 10 == 0:
            end = time.time()
            print(
                f"Iteration {i}: Loss/Loss-0 = {func(flm):.5f}/{loss_0:.5f}, Momentum = {momentum}, Time = {end - start:.2f} s")
            loss_history.append(func(flm))

    ##### Store outputs
    print('\n ============ Store outputs ===============')
    if reality:  # Get the full flm
        flm_full_target = sphlib.make_flm_full(Ilm, L)
        flm_full_start = sphlib.make_flm_full(flm_start, L)
        flm_full_end = sphlib.make_flm_full(flm, L)
    else:
        flm_full_target = Ilm
        flm_full_start = flm_start
        flm_full_end = flm

    # Save the flm and the loss
    np.save(args.save_dir + 'target_flm.npy', flm_full_target)
    np.save(args.save_dir + 'initial_flm.npy', flm_full_start)
    np.save(args.save_dir + 'output_flm.npy', flm_full_end)
    np.save(args.save_dir + 'loss.npy', loss_history)

    print('\n ============ END ===============')
