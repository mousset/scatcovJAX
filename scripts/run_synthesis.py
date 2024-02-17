# Specify CUDA device
from jax import jit, config
config.update("jax_enable_x64", True)

# Check we're running on GPU
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import os
import time
import argparse
import numpy as np
import jax.numpy as jnp
import optax
import pickle

import scatcovjax.Sphere_lib as sphlib
import scatcovjax.Synthesis_lib as synlib
import scatcovjax.Fast_scattering_lib as scatlib
from s2wav.filter_factory.filters import filters_directional_vectorised

import s2fft
import s2wav

"""
Script to run a synthesis. To run it do:
    $ python run_synthesis.py --L --N --Jmin --epochs --nreals --save_dir

"""

########## ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument('-sky', '--sky', help='Sky map to use.', default='LSS', type=str)
parser.add_argument('-l', '--L', help='L max value', default=32, type=int)
parser.add_argument('-n', '--N', help='N value', default=3, type=int)
parser.add_argument('-j', '--Jmin', help='J_min value', default=1, type=int)
parser.add_argument('-la', '--lam', help='lam value', default=2, type=float)
parser.add_argument('-e', '--epochs', help='Number of epochs', default=10, type=int)
parser.add_argument('-r', '--nreals', help='Number of realisations', default=1, type=int)
parser.add_argument('-s', '--save_dir', help='Path where outputs are saved.', default='./', type=str)

args = parser.parse_args()

# If save_dir does not exist we create the directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

sky = args.sky
L = args.L
N = args.N
J_min = args.Jmin
lam = args.lam
nreals = args.nreals
J_max = s2wav.utils.shapes.j_max(L)
J = J_max - J_min + 1

########## DEFAULT PARAMETERS
sampling = "mw"
multiresolution = True
reality = True
if reality:
    M = L
else:
    M = 2 * L - 1


###### LOSS FUNCTION
@jit
def loss_func_all(flm_float):
    # Make complex flm
    flm = flm_float[0, :, :] + 1j * flm_float[1, :, :]

    mean_new, var_new, S1_new, P00_new, C01_new, C11_new = scatlib.scat_cov_dir(flm, L, N, J_min, lam, sampling,
                                                                                None, reality, multiresolution,
                                                                                for_synthesis=True,
                                                                                normalisation=norm,
                                                                                filters=filters,
                                                                                quads=weights, precomps=precomps)
    _, P00prime_new = scatlib.get_P00prime(flm, filters_prime, normalisation=tP00prime_norm)

    # Control for mean + var
    loss = synlib.chi2(tmean, mean_new)
    loss += synlib.chi2(tvar, var_new)

    # Add S1, P00, C01, C11 losses
    loss += synlib.chi2(tS1, S1_new)
    loss += synlib.chi2(tP00, P00_new)
    loss += synlib.chi2(tC01, C01_new)
    loss += synlib.chi2(tC11, C11_new)

    # Add constrain on the PS with P00'
    loss += synlib.chi2(tP00prime, P00prime_new)

    return loss


#######################################################################
if __name__ == "__main__":
    start_main = time.time()
    print('\n============ Parameters ===============')
    print(f'{sky=}, {L=}, {N=}, {J=}, {J_min=}, {J_max=}, {lam=}, {reality=}, {nreals=}')

    print('\n============ Build the filters ===============')
    ### Basic filters
    filters = filters_directional_vectorised(L, N, J_min, lam)[0]
    ### Additional filters for P00'
    filters_prime = filters_directional_vectorised(L=L, N=N, J_min=J_min, lam=np.sqrt(2.5))[0][:-2, :, L - 1]

    print('\n============ Quadrature weihts ===============')
    weights = scatlib.quadrature(L, J_min, lam, sampling, None, multiresolution)

    print('\n============ Wigner precomputes ===============')
    ## Make the precomputation and store the coeff
    # Full precomputation
    precomps = scatlib.generate_full_precompute(L=L,
                                        N=N,
                                        J_min=J_min,
                                        lam=lam,
                                        sampling=sampling,
                                        reality=reality,
                                        multiresolution=multiresolution,
                                        nside=None)

    # file_to_store = open(f"/travail/lmousset/precomp/precomps_L{L}_N{N}.pickle", "wb")
    # pickle.dump(precomps, file_to_store)
    # file_to_store.close()

    ## Load the coeffs previously computed
    #file_to_read = open(f"/travail/lmousset/precomp/precomps_L{L}_N{N}.pickle", "rb")
    #precomps = pickle.load(file_to_read)
    #file_to_read.close()

    print('\n============ Make the target map ===============')
    print(f'Target = {sky}')
    ### SKY
    if sky == 'lss':
        repo = '/travail/lmousset/CosmoGrid/CosmoFiducial_barionified_nside512/'
        f_target, flm_target = sphlib.make_CosmoGrid_sky(L, dirmap=repo, run=0, idx_z=10, sampling=sampling,
                                                        nest=False, normalize=True, reality=reality)
    elif sky == 'cmb':
        f_target, flm_target = sphlib.make_pysm_sky(L, 'cmb', sampling=sampling, nest=False,
                                                    normalize=True, reality=reality)
    elif sky == 'tsz':
        mapfile = '/travail/lmousset/NASAsimu/tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits'
        f_target, flm_target = sphlib.make_NASAsimu_sky(L, mapfile=mapfile, sampling=sampling,
                                                        nest=False, normalize=True, reality=reality)
    elif sky == 'lensing':
        mapfile = '/travail/lmousset/Ulagam/kappa_00099.fits'
        f_target, flm_target = sphlib.make_NASAsimu_sky(L, mapfile=mapfile, sampling=sampling,
                                                        nest=False, normalize=True, reality=reality)
    elif sky == 'venus':
        f_target, flm_target = sphlib.make_planet(L, 'venus', normalize=True, reality=reality)
    else:
        raise ValueError('sky must be in lss, cmb, tsz, lensing, venus.')

    print('\n============ Compute target statistics ===============')
    ### Make a trianguler mask for alm in the 2D array
    themask = np.real(np.nan_to_num(flm_target / flm_target))  # [L, M]

    ### Power spectrum
    ps_target = sphlib.compute_ps(flm_target)

    ### STD des flm de la target
    concat = np.concatenate((np.real(flm_target), np.imag(flm_target)))
    sigma_bar = np.std(concat[concat != 0])
    print('Sigma_bar target = ', sigma_bar)

    ### P00prime used for normalisation
    tP00prime_ell, tP00prime_norm = scatlib.get_P00prime(flm_target, filters_prime, normalisation=None)
    # By contruction, P00' are 1 for the target
    _, tP00prime = scatlib.get_P00prime(flm_target, filters_prime, normalisation=tP00prime_norm)

    ### P00 for normalisation
    tP00_norm = scatlib.get_P00only(flm_target, L, N, J_min, lam, sampling, None,
                                    reality, multiresolution, for_synthesis=False, normalisation=None,
                                    filters=filters, quads=weights, precomps=precomps)  # [J][Norient]
    norm = tP00_norm

    ### Scat coeffs S1, P00, C01, C11
    # tP00 is one by definition of the normalisation (if norm is not None)
    tcoeffs = scatlib.scat_cov_dir(flm_target, L, N, J_min, lam, sampling, None,
                                   reality, multiresolution, for_synthesis=True, normalisation=norm,
                                   filters=filters, quads=weights, precomps=precomps)
    tmean, tvar, tS1, tP00, tC01, tC11 = tcoeffs  # 1D arrays

    print(f'\n ============ Store outputs for the target ===============')
    np.save(args.save_dir + 'flm_target.npy', flm_target)
    np.save(args.save_dir + 'coeffs_target.npy', np.array(tcoeffs, dtype=object), allow_pickle=True)

    ### Clean memory (probably useless)
    del(f_target)
    del(flm_target)

    ### Start the optimization
    for r in range(nreals):
        print(f'\n============ START REAL {r+1}/{nreals} ===============')
        start_real = time.time()
        print('\n============ Build initial conditions ===============')
        ### Make white noise alm
        flm_imag = np.random.randn(L, M) * sigma_bar * themask
        flm_real = np.random.randn(L, M) * sigma_bar * themask
        flm = flm_real + 1j * flm_imag

        ### Save the start point as we will iterate on flm
        flm_start = jnp.copy(flm)
        np.save(args.save_dir + f'flm_start_{r}.npy', flm)

        #Compute again coefficients of start
        scoeffs = scatlib.scat_cov_dir(flm, L, N, J_min, lam, sampling, None,
                               reality, multiresolution, for_synthesis=True, normalisation=norm,
                               filters=filters, quads=weights, precomps=precomps)
        np.save(args.save_dir + f'coeffs_start_{r}.npy', np.array(scoeffs, dtype=object), allow_pickle=True)
        # Make a float array for the optimizer
        flm_float = jnp.array([jnp.real(flm), jnp.imag(flm)])  # [2, L, L]

        print("\n============ Start the synthesis on all coeffs including P00' ===============")
        print('Using LBFGS from jaxopt.ScipyMinimize')
        flm, loss_history = synlib.fit_jaxopt_Scipy(flm_float, loss_func_all, method='L-BFGS-B', niter=args.epochs,
                                                     loss_history=None)

        ### Rebuild the complex flm array
        flm_end = flm[0, :, :] + 1j * flm[1, :, :]

        print('\n============ Compute again coefficients of end  ===============')
        ecoeffs = scatlib.scat_cov_dir(flm_end, L, N, J_min, lam, sampling, None,
                               reality, multiresolution, for_synthesis=True, normalisation=norm,
                               filters=filters, quads=weights, precomps=precomps)

        print(f'\n ============ Store outputs for real {r} ===============')
        # Save the flm and the loss
        np.save(args.save_dir + f'flm_end_{r}.npy', flm_end)
        np.save(args.save_dir + f'loss_{r}.npy', loss_history)
        # Save the coefficients
        np.save(args.save_dir + f'coeffs_end_{r}.npy', np.array(ecoeffs, dtype=object), allow_pickle=True)

        end_real = time.time()
        print(f'Duration for real {r+1} = {end_real - start_real:.3f} s = {(end_real - start_real)/60:.3f} min')
    end_main = time.time()
    print('\n ============ END ===============')
    print(f'=== Main took {(end_main - start_main)/60:.3f} min = {(end_main - start_main)/3600:.3f} h ===')
