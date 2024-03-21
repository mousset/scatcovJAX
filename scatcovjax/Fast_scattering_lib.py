from jax import jit
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from typing import List

import scatcovjax.Sphere_lib as sphlib
import s2wav
from s2wav.transforms import jax_wavelets_precompute as wavelets

import s2fft
from s2fft.precompute_transforms.spherical import forward_transform_jax
from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax


def get_P00prime(flm, filter_lin, normalisation=None):
    P00prime_ell = jnp.sum(jnp.abs(flm[None, :, :] * filter_lin[:, :, None])**2, axis=2)  # [Nfilters, L]
    P00prime = jnp.mean(P00prime_ell, axis=1)  # [Nfilters]
    if normalisation is not None:
        P00prime /= normalisation
    return P00prime_ell, P00prime


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def get_P00only(
    Ilm_in: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    for_synthesis: bool = False,
    filters: jnp.ndarray = None,
    normalisation: jnp.ndarray = None,
    quads: List[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> List[jnp.ndarray]:
    J_max = s2wav.utils.shapes.j_max(L, lam=lam)  # Maximum scale
    # J = J_max - J_min + 1  # Number of scales used
    # !!! Whatever is J_min, the shape of filter is [J_max+1, L, 2L-1] (starting from 0, not J_min)

    # Quadrature weights to make spherical integral
    # They can be computed outside and pass to the function as an argument (avoid many computations.)
    if quads is None:
        quads = []
        for j in range(J_min, J_max + 1):
            Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, lam, multiresolution)
            quads.append(
                s2fft.quadrature_jax.quad_weights(Lj, sampling, nside)
            )  # [J][Lj]

    # Part of the Wigner transform that can be computed just once.
    # It can be computed outside and pass to the function as an argument (avoid many computations.)
    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    # If the map is real (only m>0 stored) so we create the (m<0) part.
    if reality:
        Ilm = sphlib.make_flm_full(Ilm_in, L)  # [L, 2L-1]
    else:
        Ilm = Ilm_in  # [L, 2L-1]

    ### Perform first (full-scale) wavelet transform W_j2 = I * Psi_j2
    W = wavelets.flm_to_analysis(
        flm=Ilm,
        L=L,
        N=N,
        J_min=J_min,
        lam=lam,
        sampling=sampling,
        nside=nside,
        reality=reality,
        multiresolution=multiresolution,
        filters=filters,
        precomps=precomps,
    )  # [J2][Norient2, Nthetaj2, Nphij2]=[J][2N-1, Lj, 2Lj-1]

    P00 = []
    for j2 in range(J_min, J_max + 1):
        # Subsampling: the resolution in the plane (l, m) is adapted at each scale j2
        # Lj2 = s2wav.utils.shapes.wav_j_bandlimit(L, j2, lam, multiresolution)  # Band limit at resolution j2

        ### Compute P00_j2 = < |W_j2(theta, phi)|^2 >_tp
        # Average over theta phi with quadrature weights
        val = jnp.sum(
            (jnp.abs(W[j2 - J_min]) ** 2) * quads[j2 - J_min][None, :, None],
            axis=(-1, -2),
        ) / (
            4 * np.pi
        )  # [Norient2]
        # val = jnp.sum((jnp.abs(W[j2 - J_min]) ** 2), axis=(-1, -2)) / (4 * np.pi)  # [Norient2]
        # val = jnp.mean((jnp.abs(W[j2 - J_min]) ** 2)
        #                * quads[j2 - J_min][None, :, None], axis=(-1, -2))  # [Norient2] mean() et non sum/4pi
        P00.append(val)  # [J2][Norient2]

    ### Normalize P00
    if normalisation is not None:
        for j2 in range(J_min, J_max + 1):
            P00[j2 - J_min] /= normalisation[j2 - J_min]

    ### Make 1D jnp arrays instead of list (for synthesis)
    if for_synthesis:
        P00 = jnp.concatenate(P00)  # [NP00] = [Norient x J]

    return P00


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
def scat_cov_dir(
    Ilm_in: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    lam: float = 2.0,
    delta_j: int = None,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    for_synthesis: bool = False,
    filters: jnp.ndarray = None,  # [J_max+1, L, 2L-1]
    normalisation: List[jnp.ndarray] = None,  # [J][Norient]
    quads: List[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> List[jnp.ndarray]:
    
    J_max = s2wav.utils.shapes.j_max(L, lam=lam)  # Maximum scale
    # J = J_max - J_min + 1  # Number of scales used
    # !!! Whatever is J_min, the shape of filter is [J_max+1, L, 2L-1] (starting from 0, not J_min)

    # If None, we compute all the coeffs possible
    if delta_j is None:
        delta_j = J_max + 1

    # Quadrature weights to make spherical integral
    # They can be computed outside and pass to the function as an argument (avoid many computations.)
    if quads is None:
        quads = []
        for j in range(J_min, J_max + 1):
            Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, lam, multiresolution)
            quads.append(
                s2fft.quadrature_jax.quad_weights(Lj, sampling, nside)
            )  # [J][Lj]

    # Part of the Wigner transform that can be computed just once.
    # It can be computed outside and pass to the function as an argument (avoid many computations.)
    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    # If the map is real (only m>0 stored), we create the (m<0) part.
    if reality:
        Ilm = sphlib.make_flm_full(Ilm_in, L)  # [L, 2L-1]
    else:
        Ilm = Ilm_in  # [L, 2L-1]

    ### Mean and Variance
    mean = jnp.abs(Ilm[0, L - 1] / (2 * jnp.sqrt(jnp.pi)))  # Take the I_00
    # Compute |Ilm|^2 = Ilm x Ilm*
    Ilm_square = Ilm * jnp.conj(Ilm)
    # Compute the variance : Sum all except the (l=0, m=0) term
    # Todo: TEST
    var = (jnp.sum(Ilm_square) - Ilm_square[0, L - 1]) / (4 * np.pi)
    # var = jnp.mean(Ilm_square - Ilm_square[0, L - 1])  # mean() et non sum()/4pi
    # var = jnp.mean(jnp.abs(Ilm[1:]) ** 2)  # Comme avant

    ### Perform first (full-scale) wavelet transform W_j2 = I * Psi_j2
    W = wavelets.flm_to_analysis(
        flm=Ilm,
        L=L,
        N=N,
        J_min=J_min,
        lam=lam,
        sampling=sampling,
        nside=nside,
        reality=reality,
        multiresolution=multiresolution,
        filters=filters,
        precomps=precomps,
    )  # [J2][Norient2, Nthetaj2, Nphij2]=[J][2N-1, Lj, 2Lj-1]

    # Initialize S1, P00, Njjprime: will be list of len J
    S1 = []
    P00 = []
    Njjprime = []
    for j2 in range(J_min, J_max + 1):
        # j2 = Jmin is only for S1 and P00 computation
        # Subsampling: the resolution in the plane (l, m) is adapted at each scale j2
        Lj2 = s2wav.utils.shapes.wav_j_bandlimit(
            L, j2, lam, multiresolution
        )  # Band limit at resolution j2

        def modulus_step_for_j(n, args):
            """
            Compute M_lm = |W|_lm for one orientation.
            This function is re-defined at each step j2.
            """
            M_lm = args
            M_lm = M_lm.at[n].add(
                forward_transform_jax(
                    f=jnp.abs(W[j2 - J_min][n]),
                    kernel=precomps[0][j2 - J_min],
                    L=Lj2,
                    sampling=sampling,
                    reality=reality,
                    spin=0,
                    nside=nside,
                )
            )
            return M_lm  # [Lj2, Mj2] = [Lj2, 2Lj2-1]

        ### Compute M_lm for all orientations
        # Initialization
        M_lm_j2 = jnp.zeros(
            (2 * N - 1, Lj2, 2 * Lj2 - 1), dtype=jnp.complex128
        )  # [Norient2, Lj2, Mj2]
        # Loop on orientations
        M_lm_j2 = lax.fori_loop(
            0, 2 * N - 1, modulus_step_for_j, M_lm_j2
        )  # [Norient2, Lj2, Mj2]

        ### Compute S1_j2 = < M >_j2
        # Take the value at (l=0, m=0) which corresponds to indices (0, Lj2-1)
        val = M_lm_j2[:, 0, Lj2 - 1] / (2 * jnp.sqrt(jnp.pi))  # [Norient2]
        # Discard the imaginary part
        val = jnp.real(val)
        S1.append(val)  # [J2][Norient2]

        ### Compute P00_j2 = < |W_j2(theta, phi)|^2 >_tp
        # Average over theta phi (Parseval)
        val = jnp.sum(
            (jnp.abs(W[j2 - J_min]) ** 2) * quads[j2 - J_min][None, :, None],
            axis=(-1, -2),
        ) / (
            4 * np.pi
        )  # [Norient2]
        # Other way: average over lm (Parseval) : P00_j2 = < |M_lm|^2 >_j2 (does not give exactly the same)
        # val = jnp.sum(jnp.abs(M_lm_j2) ** 2, axis=(-1, -2)) / (4 * np.pi)  # [Norient2]
        # Todo: TEST
        # val = jnp.mean((jnp.abs(W[j2 - J_min]) ** 2)
        #               * quads[j2 - J_min][None, :, None], axis=(-1, -2))  # [Norient2] mean() et non sum/4pi

        # val = jnp.mean(jnp.abs(W[j2 - J_min]) ** 2, axis=(-1, -2)) * Lj2/L # Comme avant autre version
        # val = jnp.mean(jnp.abs(M_lm_j2) ** 2, axis=(-1, -2)) * Lj2/L # Comme avant
        # val = jnp.sum((jnp.abs(W[j2 - J_min]) ** 2), axis=(-1, -2)) / (4 * np.pi)  # [Norient2]
        P00.append(val)  # [J2][Norient2]

        ### Compute Njjprime
        # The iteration at j2=Jmin was only needed for S1 and P00 so we do not do the Njjprime computation when j2=Jmin.
        if j2 != J_min:
            # Filters: We must keep all scales
            # the selection from J_min to J_max=j2-1 is done in the function flm_to_analysis()
            filters_j2 = filters[:, :Lj2, L - Lj2 : L - 1 + Lj2]
            ### Compute Njjprime
            Njjprime_for_j2 = []
            # TODO: This loop will increase compile time.
            for n in range(2 * N - 1):  # Loop on orientations
                # Wavelet transform of Mlm: Nj1j2 = M_j2 * Psi_j1
                # val shape is [J1j][Norient1, Nthetaj1, Nphij1]
                # Not sure of the len of val (some terms are 0)
                val = wavelets.flm_to_analysis(
                    flm=M_lm_j2[n],
                    L=Lj2,
                    N=N,
                    J_min=J_min,
                    J_max=j2 - 1,  # Only do convolutions at larger scales: from J_min to j2-1
                    lam=lam,
                    sampling=sampling,
                    nside=nside,
                    reality=reality,
                    multiresolution=multiresolution,
                    filters=filters_j2,
                    precomps=[
                        0,
                        0,
                        precomps[2][: (j2 - 1) - J_min + 1],
                    ],  # precomps are ordered from J_min to J_max
                )  # [J1][Norient1, Nthetaj1, Nphij1]
                Njjprime_for_j2.append(
                    val
                )  # [Norient2][Jj1][Norient1, Nthetaj1, Nphij1]
            Njjprime.append(
                Njjprime_for_j2
            )  # [J2-1][Norient2][J1j][Norient1, Nthetaj, Nphij] (M_j2 * Psi_j1)

    ### Reorder and flatten Njjprime, convert to JAX arrays for C01/C11
    Njjprime_flat = []
    # For C01 and C11, we need j1 < j2 so j1=Jmax is not possible, this is why j1 goes from J_min to J_max-1.
    for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1
        Njjprime_flat_for_j2 = []
        for j2 in range(j1 + 1, min(j1 + delta_j, J_max + 1)):  # j1+1 <= j2 <= J_max
            Njjprime_flat_for_n2 = []
            for n2 in range(2 * N - 1):
                # In Njjprime, j2 starts at Jmin + 1 while j1 starts at Jmin
                Njjprime_flat_for_n2.append(
                    Njjprime[j2 - J_min - 1][n2][j1 - J_min][:, :, :]
                )  # [Norient2][Norient1, Nthetaj1, Nphij1]
            Njjprime_flat_for_j2.append(
                Njjprime_flat_for_n2
            )  # [J2][Norient2][Norient1, Nthetaj1, Nphij1]
        Njjprime_flat.append(
            jnp.array(Njjprime_flat_for_j2)
        )  # [J1][J2, Norient2, Norient1, Nthetaj1, Nphij1]

    ### Compute C01 and C11
    # Indexing: a/b = j3/j2, j/k = n3/n2, n = n1, theta = t, phi = p
    C01 = []
    C11 = []
    for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1
        ### Compute C01
        # C01 = <W_j1 x (M_j3 * Psi_j1)*> = <W_j1 x (N_j1j3)*> so we must have j1 < j3
        # Do the product W_j1n1tp x N_j1j3n3n1tp*
        val = jnp.einsum(
            "ajntp,ntp->ajntp",
            jnp.conj(Njjprime_flat[j1 - J_min]),
            W[j1 - J_min],
            optimize=True,
        )  # [J3_j1, Norient3, Norient1, Nthetaj1, Nphij1]
        # Average over Theta and Phi: <val>_j3n3n1 = Sum_tp (val_j3n3n1tp x quad_t) / 4pi
        val = jnp.einsum(
            "ajntp,t->ajn", val, quads[j1 - J_min], optimize=True
        )  # [J3_j1, Norient3, Norient1]
        # val /= (4 * np.pi)
        # Discard the imaginary part
        val = jnp.real(val)
        C01.append(val)  # [J1-1][J3_j1, Norient3, Norient1]

        ### Compute C11
        # C11 = <(M_j3 * Psi_j1) x (M_j2 * Psi_j1)*> = <N_j1j3 x N_j1j2*> we have j1 < j2 <= j3
        # Do the product N_j1j3n3n1tp x N_j1j2n2n1tp*
        val = Njjprime_flat[j1 - J_min]  # [J2_j1, Norient2, Norient1, Nthetaj1, Nphij1]
        val = jnp.einsum(
            "ajntp,bkntp->abjkntp", val, jnp.conj(val), optimize=True
        )  # [J3_j1, J2_j1, Norient3, Norient2, Norient1, Nthetaj1, Nphij1]
        # Average over Theta and Phi: <val>_j3j2n3n2n1 = Sum_tp (val_j3j2n3n2n1tp x quad_t) / 4pi
        val = jnp.einsum(
            "abjkntp,t->abjkn", val, quads[j1 - J_min], optimize=True
        )  # [J3_j1, J2_j1, Norient3, Norient2, Norient1]
        # val /= (4 * np.pi)
        # Discard the imaginary part
        val = jnp.real(val)
        C11.append(val)  # [J1-1][J3_j1, J2_j1, Norient3, Norient2, Norient1]

    ### Normalize the coefficients
    if normalisation is not None:
        ### S1 and P00
        for j2 in range(J_min, J_max + 1):  # J_min <= j2 <= J_max
            S1[j2 - J_min] /= jnp.sqrt(normalisation[j2 - J_min])
            P00[j2 - J_min] /= normalisation[j2 - J_min]
        ## C01 and C11
        for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1
            C01[j1 - J_min] = jnp.einsum(
                "ajn,j->ajn",
                C01[j1 - J_min],
                1 / jnp.sqrt(normalisation[j1 - J_min]),
                optimize=True,
            )
            C01[j1 - J_min] = jnp.einsum(
                "ajn,n->ajn",
                C01[j1 - J_min],
                1 / jnp.sqrt(normalisation[j1 - J_min]),
                optimize=True,
            )

            C11[j1 - J_min] = jnp.einsum(
                "abjkn,j->abjkn",
                C11[j1 - J_min],
                1 / jnp.sqrt(normalisation[j1 - J_min]),
                optimize=True,
            )
            C11[j1 - J_min] = jnp.einsum(
                "abjkn,k->abjkn",
                C11[j1 - J_min],
                1 / jnp.sqrt(normalisation[j1 - J_min]),
                optimize=True,
            )

    ### Make 1D jnp arrays instead of list (for synthesis)
    if for_synthesis:
        S1 = jnp.concatenate(S1)  # [NS1] = [Norient x J]
        P00 = jnp.concatenate(P00)  # [NP00] = [Norient x J]
        C01 = jnp.concatenate(C01, axis=None)  # [NC01]
        C11 = jnp.concatenate(C11, axis=None)  # [NC11]

        # !!! TEST
        # S1 = jnp.log(S1)
        # P00 = jnp.log(P00)

    return mean, var, S1, P00, C01, C11


def quadrature(
    L: int,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = True,
):
    J = s2wav.utils.shapes.j_max(L, lam=lam)
    quads = []
    for j in range(J_min, J + 1):
        Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, lam, multiresolution)
        #quads.append(s2fft.utils.quadrature_jax.quad_weights_mw_theta_only(Lj))
        quads.append(s2fft.utils.quadrature_jax.quad_weights(Lj, sampling, nside))
    return quads


def generate_full_precompute(
    L: int,
    N: int,
    J_min: int,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = True,
    reality: bool = False
):
    J_max = s2wav.utils.shapes.j_max(L, lam=lam)  # Maximum scale
    precomps = wavelets.generate_precomputes(
        L=L,
        N=N,
        J_min=J_min,
        lam=lam,
        forward=False,
        reality=reality,
        multiresolution=multiresolution,
        nospherical=True,
    )
    for j2 in range(J_min, J_max + 1):
        # j2 = Jmin is only for S1 and P00 computation
        # Subsampling: the resolution in the plane (l, m) is adapted at each scale j2
        Lj2 = s2wav.utils.shapes.wav_j_bandlimit(
            L, j2, lam, multiresolution
        )  # Band limit at resolution j2
        precomps[0].append(
            spin_spherical_kernel_jax(
                L=Lj2,
                spin=0,
                reality=reality,
                sampling=sampling,
                nside=nside,
                forward=True,
            )
        )
    return precomps
