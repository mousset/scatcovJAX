# Specify CUDA device
from jax import jit, config
config.update("jax_enable_x64", True)

# Check we're running on GPU
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import os
import argparse
import numpy as np
import jax.numpy as jnp
import optax

import scatcovjax.Sphere_lib as sphlib
import scatcovjax.Synthesis_lib as synlib
from scatcovjax.Scattering_lib import scat_cov_axi, scat_cov_dir, quadrature
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
normalisation = None
reality = True
J_min = 0
momentum = 2.  # For naive gradient descent
momentum_ps_only = 0.1
learning_rate = 1e-1  # For optax gradient descent
learning_rate_ps_only = 1e-3
epochs_ps_only = 2
planet = 'venus'

###### Loss functions
@jit
def loss_func_ps_only(flm):
    ps = jnp.sum(jnp.abs(flm) ** 2, axis=-1)
    return jnp.sum(jnp.abs(ps - ps_target) ** 2)

@jit
def loss_func(flm):
    if axi:
        mean_new, var_new, S1_new, P00_new, C01_new, C11_new = scat_cov_axi(flm, L, N, J_min, sampling,
                                                                            None, reality, multiresolution,
                                                                            normalisation, filters=filters)
    else:
        mean_new, var_new, S1_new, P00_new, C01_new, C11_new = scat_cov_dir(flm, L, N, J_min, sampling,
                                                                            None, reality, multiresolution,
                                                                            normalisation, filters=filters,
                                                                            quads=weights, precomps=precomps)
    # Control for mean + var
    loss = synlib.chi2(tmean, mean_new)
    loss += synlib.chi2(tvar, var_new)

    # Add S1, P00, C01, C11 losses
    loss += synlib.chi2(tS1, S1_new)
    loss += synlib.chi2(tP00, P00_new)
    loss += synlib.chi2(tC01, C01_new)
    loss += synlib.chi2(tC11, C11_new)

    return loss


######################################################################################


if __name__ == "__main__":
    print('\n============ Parameters ===============')
    print(f'{axi=} , {L=}, {N=}, {J=}, {J_min=}, {reality=}, {planet=}')

    print('\n============ Build the filters ===============')
    filters = filters_directional_vectorised(L, N, J_min)[0]

    print('\n============ Quadrature weihts ===============')
    weights = quadrature(L, J_min, sampling, None, multiresolution)

    print('\n============ Wigner precomputes ===============')
    precomps = s2wav.transforms.jax_wavelets.generate_wigner_precomputes(L, N, J_min, 2.0, sampling, None, False,
                                                                         reality, multiresolution)

    print('\n============ Make the target ===============')
    f_target, flm_target = sphlib.make_MW_lensing(L, normalize=True, reality=reality)
    # f_target, flm_target = sphlib.make_MW_planet(L, planet, normalize=True, reality=reality)

    # Power spectrum of the target
    ps_target = sphlib.compute_ps(flm_target)

    # Scat coeffs S1, P00, C01, C11
    if axi:
        tcoeffs = scat_cov_axi(flm_target, L, N, J_min, sampling, None,
                                reality, multiresolution, normalisation, filters=filters)
    else:
        tcoeffs = scat_cov_dir(flm_target, L, N, J_min, sampling, None,
                               reality, multiresolution, normalisation,
                               filters=filters, quads=weights, precomps=precomps)
    tmean, tvar, tS1, tP00, tC01, tC11 = tcoeffs

    print('\n============ Build initial conditions ===============')
    # Gaussian white noise in pixel space
    f = np.random.randn(L, 2 * L - 1).astype(np.float64)
    flm = s2fft.forward_jax(f, L, reality=reality)

    # Gaussian white noise in flm space TODO: This is wrong
    # flm = np.random.randn(L, 2 * L - 1).astype(np.float64) + 1j * np.random.randn(L, 2 * L - 1).astype(np.float64)

    flm = flm[:, L - 1:] if reality else flm

    flm_start = jnp.copy(flm)  # Save the start point as we will iterate on flm

    print('\n============ Start the synthesis on PS only ===============')
    # Naive implementation
    flm, loss_history = synlib.fit_brutal(flm, loss_func_ps_only,
                                          momentum=momentum_ps_only, niter=epochs_ps_only, loss_history=None)
    # Optax implementation
    # optimizer = optax.adam(learning_rate_ps_only)
    # flm, loss_history = synlib.fit_optax(flm, optimizer, loss_func_ps_only, niter=epochs_ps_only, loss_history=None)

    print('\n============ Start the synthesis on all coeffs ===============')
    # Naive implementation
    flm_end, loss_history = synlib.fit_brutal(flm, loss_func,
                                              momentum=momentum, niter=args.epochs, loss_history=loss_history)
    # Optax implementation
    # optimizer = optax.adam(learning_rate)
    # flm_end, loss_history = synlib.fit_optax(flm, optimizer, loss_func, niter=args.epochs, loss_history=loss_history)

    print('\n============ Compute again coefficients of start and end  ===============')
    if axi:
        scoeffs = scat_cov_axi(flm_start, L, N, J_min, sampling, None,
                                reality, multiresolution, normalisation, filters=filters)
        ecoeffs= scat_cov_axi(flm_end, L, N, J_min, sampling, None,
                              reality, multiresolution, normalisation, filters=filters)
    else:
        scoeffs = scat_cov_dir(flm_start, L, N, J_min, sampling, None,
                               reality, multiresolution, normalisation,
                               filters=filters, quads=weights, precomps=precomps)
        ecoeffs = scat_cov_dir(flm_end, L, N, J_min, sampling, None,
                               reality, multiresolution, normalisation,
                               filters=filters, quads=weights, precomps=precomps)

    print('\n ============ Store outputs ===============')
    # Save the flm and the loss
    np.save(args.save_dir + 'flm_target.npy', flm_target)
    np.save(args.save_dir + 'flm_start.npy', flm_start)
    np.save(args.save_dir + 'flm_end.npy', flm_end)
    np.save(args.save_dir + 'loss.npy', loss_history)

    # Save the coefficients
    np.save(args.save_dir + 'coeffs_target.npy', tcoeffs)
    np.save(args.save_dir + 'coeffs_start.npy', scoeffs)
    np.save(args.save_dir + 'coeffs_end.npy', ecoeffs)

    print('\n ============ END ===============')
