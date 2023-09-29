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
from scatcovjax.Scattering_lib import scat_cov_dir, quadrature, get_P00only
from s2wav.filter_factory.filters import filters_directional_vectorised

import s2fft
import s2wav

"""
Script to run a synthesis. To run it do:
    $ python run_synthesis.py --L --N --Jmin --epochs --save_dir

"""

########## ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument('-l', '--L', help='L max value', default=32, type=int)
parser.add_argument('-n', '--N', help='N value', default=3, type=int)
parser.add_argument('-j', '--Jmin', help='J_min value', default=1, type=int)
parser.add_argument('-e', '--epochs', help='Number of epochs', default=10, type=int)
parser.add_argument('-r', '--nreals', help='Number of realisations', default=1, type=int)
parser.add_argument('-s', '--save_dir', help='Path where outputs are saved.', default='./', type=str)

args = parser.parse_args()

# If save_dir does not exist we create the directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

L = args.L
N = args.N
J_min = args.Jmin
nreals = args.nreals
J_max = s2wav.utils.shapes.j_max(L)
J = J_max - J_min + 1

########## DEFAULT PARAMETERS
sampling = "mw"
multiresolution = True
reality = True


###### Loss functions
@jit
def loss_func_P00_only(flm_float):
    flm = flm_float[0, :, :] + 1j * flm_float[1, :, :]

    P00_new = get_P00only(flm, L, N, J_min, sampling,
                                  None, reality, multiresolution, for_synthesis=True,
                                  normalisation=norm, filters=filters,
                                  quads=weights, precomps=precomps)
    loss = synlib.chi2(tP00, P00_new)
    return loss


@jit
def loss_func(flm_float):
    # Make complex flm
    flm = flm_float[0, :, :] + 1j * flm_float[1, :, :]

    mean_new, var_new, S1_new, P00_new, C01_new, C11_new = scat_cov_dir(flm, L, N, J_min, sampling,
                                                                                None, reality, multiresolution,
                                                                                for_synthesis=True,
                                                                                normalisation=norm,
                                                                                filters=filters,
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


#######################################################################
if __name__ == "__main__":
    print('\n============ Parameters ===============')
    print(f'{L=}, {N=}, {J=}, {J_min=}, {J_max=}, {reality=}, {nreals=}')

    print('\n============ Build the filters ===============')
    filters = filters_directional_vectorised(L, N, J_min)[0]

    print('\n============ Quadrature weihts ===============')
    weights = quadrature(L, J_min, sampling, None, multiresolution)

    print('\n============ Wigner precomputes ===============')
    precomps = s2wav.transforms.jax_wavelets.generate_wigner_precomputes(L, N, J_min, 2.0, sampling, None, False,
                                                                         reality, multiresolution)

    print('\n============ Make the target ===============')
    ### Sky
    repo = '/travail/lmousset/CosmoGrid/CosmoFiducial_barionified_nside512/'
    f_target, flm_target = sphlib.make_CosmoGrid_sky(L, dirmap=repo, run=0, idx_z=10, sampling=sampling,
                                                     nest=False, normalize=True, reality=reality)
    print('Target = LSS map : I = np.log(I + 0.001) - 2')

    # f_target, flm_target = sphlib.make_MW_lensing(L, normalize=True, reality=reality)
    # print('Target = LSS map')

    # f_target, flm_target = sphlib.make_pysm_sky(L, 'cmb', sampling=sampling, nest=False, normalize=True, reality=reality)
    # print('Target = CMB map')

    # f_target, flm_target = sphlib.make_planet(L, planet, normalize=True, reality=reality)
    # print('Target = Planet map')

    ### Power spectrum of the target
    ps_target = sphlib.compute_ps(flm_target)

    ### P00 for normalisation
    tP00_norm = get_P00only(flm_target, L, N, J_min, sampling, None,
                                    reality, multiresolution, for_synthesis=False, normalisation=None,
                                    filters=filters, quads=weights, precomps=precomps)  # [J][Norient]
    norm = tP00_norm

    ### Scat coeffs S1, P00, C01, C11
    # tP00 is one by definition of the normalisation (if norm is not None)
    tcoeffs = scat_cov_dir(flm_target, L, N, J_min, sampling, None,
                                   reality, multiresolution, for_synthesis=True, normalisation=norm,
                                   filters=filters, quads=weights, precomps=precomps)
    tmean, tvar, tS1, tP00, tC01, tC11 = tcoeffs  # 1D arrays

    print(f'\n ============ Store outputs for the target ===============')
    np.save(args.save_dir + 'flm_target.npy', flm_target)
    np.save(args.save_dir + 'coeffs_target.npy', tcoeffs)

    for r in range(nreals):
        print(f'\n============ START REAL {r+1}/{nreals} ===============')
        print('\n============ Build initial conditions ===============')
        # Gaussian white noise in pixel space
        print('White noise MW pixel space with the STD of the target')
        # If the target map is normalized, we should have tvar = 1.
        print(f'{tvar=}')
        # np.random.seed(42)  # Fix the seed
        if reality:  # Real map
            # f = jnp.sqrt(tvar) * np.random.randn(L, 2 * L - 1).astype(np.float64)
            f = jnp.std(f_target) * np.random.randn(L, 2 * L - 1).astype(np.float64)
        else:
            # f = jnp.sqrt(tvar) * np.random.randn(L, 2 * L - 1).astype(np.float64) + \
            #     1j * np.random.randn(L, 2 * L - 1).astype(np.float64)
            f = jnp.std(f_target) * np.random.randn(L, 2 * L - 1).astype(np.float64) + \
                1j * np.random.randn(L, 2 * L - 1).astype(np.float64)
        # SHT to make the initial flm
        flm = s2fft.forward_jax(f, L, reality=reality)

        # Cut the flm
        flm = flm[:, L - 1:] if reality else flm

        # Save the start point as we will iterate on flm
        flm_start = jnp.copy(flm)

        # Make a float array for the optimizer
        flm_float = jnp.array([jnp.real(flm), jnp.imag(flm)])  # [2, L, L]

        # print('\n============ Start the synthesis on P00 only ===============')
        # print('Using jaxopt.GradientDescent')
        # flm, loss_history = synlib.fit_jaxopt(flm_float, loss_func_P00_only, method='GradientDescent', niter=args.epochs,
        #                                       loss_history=None)
        # print('Using LBFGS from jaxopt.ScipyMinimize')
        # flm, loss_history = synlib.fit_jaxopt_Scipy(flm_float, loss_func_P00_only, method='L-BFGS-B', niter=args.epochs,
        #                                             loss_history=None)

        print('\n============ Start the synthesis on all coeffs ===============')
        print('Using jaxopt.GradientDescent')
        flm, loss_history = synlib.fit_jaxopt(flm_float, loss_func, method='GradientDescent', niter=args.epochs,
                                              loss_history=None)
        # print('Using LBFGS from jaxopt.ScipyMinimize')
        # flm, loss_history = synlib.fit_jaxopt_Scipy(flm_float, loss_func, method='L-BFGS-B', niter=args.epochs,
        #                                             loss_history=None)
        # print('Using optax.adam')
        # lr = 1e-2
        # optimizer = optax.adam(lr)
        # flm, loss_history = synlib.fit_optax(flm, optimizer, loss_func, niter=args.epochs,
        #                                      loss_history=None)
        # flm_end = jnp.copy(flm)

        ### Rebuild the complex flm array
        flm_end = flm[0, :, :] + 1j * flm[1, :, :]

        print('\n============ Compute again coefficients of start and end  ===============')
        scoeffs = scat_cov_dir(flm_start, L, N, J_min, sampling, None,
                               reality, multiresolution, for_synthesis=True, normalisation=norm,
                               filters=filters, quads=weights, precomps=precomps)
        ecoeffs = scat_cov_dir(flm_end, L, N, J_min, sampling, None,
                               reality, multiresolution, for_synthesis=True, normalisation=norm,
                               filters=filters, quads=weights, precomps=precomps)

        print(f'\n ============ Store outputs for real {r} ===============')
        # Save the flm and the loss
        np.save(args.save_dir + f'flm_start_{r}.npy', flm_start)
        np.save(args.save_dir + f'flm_end_{r}.npy', flm_end)
        np.save(args.save_dir + f'loss_{r}.npy', loss_history)
        # Save the coefficients
        np.save(args.save_dir + f'coeffs_start_{r}.npy', scoeffs)
        np.save(args.save_dir + f'coeffs_end_{r}.npy', ecoeffs)

    print('\n ============ END ===============')
