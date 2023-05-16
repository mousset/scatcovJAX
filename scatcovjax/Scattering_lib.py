from jax import jit, config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from typing import List, Tuple

import scatcovjax.Sphere_lib as sphlib
import s2wav
import s2fft


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def get_P00only(
    Ilm_in: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    normalisation: jnp.ndarray = None,
    filters: jnp.ndarray = None,
    quads: List[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> List[jnp.ndarray]:
    J = s2wav.utils.shapes.j_max(L)

    if quads is None:
        quads = []
        for j in range(J_min, J + 1):
            Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, 2.0, multiresolution)
            quads.append(s2fft.utils.quadrature_jax.quad_weights(Lj, sampling, nside))

    if precomps == None:
        precomps = s2wav.transforms.jax_wavelets.generate_wigner_precomputes(
            L, N, J_min, 2.0, sampling, nside, False, reality, multiresolution
        )

    if reality:
        Ilm = sphlib.make_flm_full(Ilm_in, L)
    else:
        Ilm = Ilm_in

    # Perform first (full-scale) wavelet transform
    W = s2wav.flm_to_analysis(
        Ilm,
        L,
        N,
        J_min,
        sampling=sampling,
        nside=nside,
        reality=reality,
        multiresolution=multiresolution,
        filters=filters,
        precomps=precomps
    )
    P00 = []
    for j in range(J_min, J + 1):
        Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, 2.0, multiresolution)

        # Compute P00
        # W[j - J_min] = [Lj, 2Lj-1]
        val = jnp.mean(jnp.abs(W[j - J_min]) ** 2, axis=(-1, -2))
        #val = jnp.sum(jnp.abs(W[j - J_min]) ** 2, axis=(-1, -2)) / (4 * np.pi)
        val *= Lj / L
        P00.append(val)

    P00 = jnp.concatenate(P00)  # [J]
    if normalisation is not None:
        P00 /= normalisation

    return P00


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def scat_cov_dir(
    Ilm_in: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    for_synthesis: bool = False,
    filters: jnp.ndarray = None,  # [J_max+1, L, 2L-1]
    normalisation: List[jnp.ndarray] = None,  # [J][Norient]
    quads: List[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None
) -> List[jnp.ndarray]:

    J_max = s2wav.utils.shapes.j_max(L)  # Maximum scale
    # J = J_max - J_min + 1  # Number of scales used
    # !!! Whatever is J_min, the shape of filter is [J_max+1, L, 2L-1] (starting from 0, not J_min)

    # Quadrature weights to make spherical integral
    # They can be computed outside and pass to the function as an argument (avoid many computations.)
    if quads is None:
        quads = []
        for j in range(J_min, J_max + 1):
            Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, 2.0, multiresolution)
            print('Lj', Lj)
            quads.append(s2fft.quadrature_jax.quad_weights(Lj, sampling, nside))  # [J][Lj]
    print('quad', len(quads), quads[0].shape, quads[1].shape)

    # Part of the Wigner transform that can be computed just once.
    # It can be computed outside and pass to the function as an argument (avoid many computations.)
    if precomps == None:
        precomps = s2wav.transforms.jax_wavelets.generate_wigner_precomputes(
            L, N, J_min, 2.0, sampling, nside, False, reality, multiresolution
        )  # [J][J_max+1][?]
    print('precomps', len(precomps), len(precomps[0]), precomps[0][0].shape, precomps[0][-1].shape)

    # If the map is real (only m>0 stored), we create the (m<0) part.
    if reality:
        Ilm = sphlib.make_flm_full(Ilm_in, L)  # [L, 2L-1]
    else:
        Ilm = Ilm_in  # [L, 2L-1]

    ### Mean and Variance
    # Todo!! Je pense que les normalisations sont ok ici
    mean = jnp.abs(Ilm[0, L - 1] / (2 * jnp.sqrt(jnp.pi)))  # Take the I_00
    # Compute |Ilm|^2 = Ilm x Ilm*
    Ilm_square = Ilm * jnp.conj(Ilm)
    # Compute the variance : Sum all except the (l=0, m=0) term
    var = (np.sum(Ilm_square) - Ilm_square[0, L - 1]) / (4 * np.pi)

    ### Perform first (full-scale) wavelet transform W_j2 = I * Psi_j2
    W = s2wav.flm_to_analysis(
        Ilm,
        L,
        N,
        J_min,
        sampling=sampling,
        nside=nside,
        reality=reality,
        multiresolution=multiresolution,
        filters=filters,
        precomps=precomps
    )  # [J2][Norient2, Nthetaj2, Nphij2]=[J][2N-1, Lj, 2Lj-1]
    print('W', len(W), W[0].shape, W[-1].shape)

    # Initialize S1, P00, Njjprime: will be list of len J
    S1 = []
    P00 = []
    Njjprime = []
    for j2 in range(J_min, J_max + 1):
        print(f'\n {j2=}')
        # Subsampling: the resolution in the plane (l, m) is adapted at each scale j2
        Lj2 = s2wav.utils.shapes.wav_j_bandlimit(L, j2, 2.0, multiresolution)  # Band limit at resolution j2
        print(f'{Lj2=}')
        Njjprime_for_j2 = []

        def modulus_step_for_j(n, args):
            """
            Compute M_lm = |W|_lm for one orientation.
            This function is re-defined at each step j2.
            """
            M_lm = args
            M_lm = M_lm.at[n].add(
                s2fft.forward_jax(
                    jnp.abs(W[j2 - J_min][n]),
                    Lj2,
                    0,
                    sampling=sampling,
                    nside=nside,
                    reality=reality,
                )
            )
            return M_lm  # [Lj2, Mj2] = [Lj2, 2Lj2-1]

        ### Compute M_lm for all orientations
        # Initialization
        M_lm_j2 = jnp.zeros((2 * N - 1, Lj2, 2 * Lj2 - 1), dtype=jnp.complex128)  # [Norient2, Lj2, Mj2]
        # Loop on orientations
        M_lm_j2 = lax.fori_loop(0, 2 * N - 1, modulus_step_for_j, M_lm_j2)  # [Norient2, Lj2, Mj2]

        ### Compute S1_j2 = < M_lm >_j2
        # Take the value at (l=0, m=0) which corresponds to indices (0, Lj2-1)
        # Todo!! A priori ok
        val = M_lm_j2[:, 0, Lj2 - 1] / (2 * jnp.sqrt(jnp.pi))  # [Norient2]
        S1.append(val)  # [J2][Norient2]

        ### Compute P00_j2 = < |M_lm|^2 >_j2
        # Average over lm (Parseval)
        # Todo!!
        # val = jnp.sum(jnp.abs(M_lm_j2) ** 2, axis=(-1, -2)) / (4 * np.pi)  # [Norient2]
        # print('P00 1st way', val)
        val = jnp.sum((jnp.abs(W[j2 - J_min]) ** 2) * quads[j2-J_min][None, :, None], axis=(-1, -2)) / (4 * np.pi)
        print('P00 2nd way', val)
        P00.append(val)  # [J2][Norient2]

        ### Compute Njjprime
        # TODO: This loop will increase compile time.
        for n in range(2 * N - 1):  # Loop on orientations
            # Wavelet transform of Mlm: Nj1j2 = M_j2 * Psi_j1
            val = s2wav.flm_to_analysis(
                M_lm_j2[n],
                Lj2,
                N,
                J_min,
                J_max=j2-1,  # Only do convolutions at larger scales: from J_min to j2-1
                sampling=sampling,
                nside=nside,
                reality=reality,
                multiresolution=multiresolution,
                filters=filters[J_min: j2, :Lj2, L - Lj2: L - 1 + Lj2],  # Select filters from J_min to j2-1
                precomps=precomps[:(j2-1)-J_min+1]  # precomps are ordered from J_min to J_max
            )  # [J1][Norient1, Nthetaj1, Nphij1]
            print('val', len(val), val[j2-1].shape)
            Njjprime_for_j2.append(val)  # [Norient2][Jj1][Norient1, Nthetaj1, Nphij1]
        print('Njjprime_for_j', len(Njjprime_for_j2), len(Njjprime_for_j2[-1]), Njjprime_for_j2[0][0].shape)
        Njjprime.append(Njjprime_for_j2)  # [J2][Norient2][J1j][Norient1, Nthetaj, Nphij] (M_j2 * Psi_j1)

    ### Reorder and flatten Njjprime, convert to JAX arrays for C01/C11

    Njjprime_flat = []
    # For C01 and C11, we need j1 < j2 so j1=Jmax is not possible, this is why j1 goes from J_min to J_max-1.
    for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1
        Njjprime_flat_for_j2 = []
        for j2 in range(j1 + 1, J_max + 1):  # j1+1 <= j2 <= J_max
            Njjprime_flat_for_n2 = []
            for n2 in range(2 * N - 1):
                Njjprime_flat_for_n2.append(Njjprime[j2 - J_min][n2][j1 - J_min][:, :, :])  # [Norient2][Norient1, Nthetaj1, Nphij1]
            Njjprime_flat_for_j2.append(Njjprime_flat_for_n2)  # [J2][Norient2][Norient1, Ntheta_j1, Nphi_j1]

        # [J1][J2, Norient2, Norient1, Nthetaj1, Nphij1] => [J-1][Jj2, Norient, Norient, Ntheta_j1, Nphi_j1]
        Njjprime_flat.append(jnp.array(Njjprime_flat_for_j2))
    print('Njjprime_flat', len(Njjprime_flat), Njjprime_flat[0].shape, Njjprime_flat[-1].shape)

    ### Compute C01 and C11
    # Indexing: a/b = j3/j2, j/k = n3/n2, n = n1, theta = t, phi = p
    C01 = []
    C11 = []
    # Todo!! Normalisation: j'ai ajouté la division par 4pi
    for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1
        ### Compute C01
        # C01 = <W_j1 x (M_j3 * Psi_j1)*> = <W_j1 x (N_j1j3)*> so we must have j1 < j3
        # Do the product W_j1n1tp x N_j1j3n3n1tp*
        val = jnp.einsum(
            "ajntp,ntp->ajntp", jnp.conj(Njjprime_flat[j1 - J_min]), W[j1 - J_min], optimize=True
        )  # [J3_j1, Norient3, Norient1, Nthetaj1, Nphij1]
        # Average over Theta and Phi: <val>_j3n3n1 = Sum_tp (val_j3n3n1tp x quad_t) / 4pi
        val = jnp.einsum("ajntp,t->ajn", val, quads[j1 - J_min], optimize=True)  # [J3_j1, Norient3, Norient1]
        val /= (4 * np.pi)
        C01.append(val)  # [J1-1][J3_j1, Norient3, Norient1]

        ### Compute C11
        # C11 = <(M_j3 * Psi_j1) x (M_j2 * Psi_j1)*> = <N_j1j3 x N_j1j2*> we have j1 < j2 <= j3
        # Do the product N_j1j3n3n1tp x N_j1j2n2n1tp*
        val = Njjprime_flat[j1 - J_min]  # [J2_j1, Norient2, Norient1, Nthetaj1, Nphij1]
        val = jnp.einsum("ajntp,bkntp->abjkntp", val, jnp.conj(val), optimize=True)  # [J3_j1, J2_j1, Norient3, Norient2, Norient1, Nthetaj1, Nphij1]
        # Average over Theta and Phi: <val>_j3j2n3n2n1 = Sum_tp (val_j3j2n3n2n1tp x quad_t) / 4pi
        val = jnp.einsum("abjkntp,t->abjkn", val, quads[j1 - J_min], optimize=True)  # [J3_j1, J2_j1, Norient3, Norient2, Norient1]
        val /= (4 * np.pi)
        C11.append(val)  # [J1-1][J3_j1, J2_j1, Norient3, Norient2, Norient1]

    ### Normalize the coefficients
    if normalisation is not None:
        ### S1 and P00
        for j2 in range(J_min, J_max + 1):
            S1[j2 - J_min] /= jnp.sqrt(normalisation[j2 - J_min])
            P00[j2 - J_min] /= normalisation[j2 - J_min]
        ## C01 and C11
        for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1
            C01[j1 - J_min] = jnp.einsum("ajn,j->ajn", C01[j1-J_min], 1 / jnp.sqrt(normalisation[j1-J_min]), optimize=True)
            C01[j1 - J_min] = jnp.einsum("ajn,n->ajn", C01[j1 - J_min], 1 / jnp.sqrt(normalisation[j1 - J_min]), optimize=True)

            C11[j1 - J_min] = jnp.einsum("abjkn,j->abjkn", C11[j1 - J_min], 1 / jnp.sqrt(normalisation[j1 - J_min]), optimize=True)
            C11[j1 - J_min] = jnp.einsum("abjkn,k->abjkn", C11[j1 - J_min], 1 / jnp.sqrt(normalisation[j1 - J_min]), optimize=True)


    ### Make 1D jnp arrays instead of list (for synthesis)
    if for_synthesis:
        S1 = jnp.concatenate(S1)  # [NS1] = [Norient x J]
        P00 = jnp.concatenate(P00)  # [NP00] = [Norient x J]
        C01 = jnp.concatenate(C01, axis=None)  # [NC01]
        C11 = jnp.concatenate(C11, axis=None)  # [NC11]
        print('S1, P00, C01, C11', S1.shape, P00.shape, C01.shape, C11.shape)

    return mean, var, S1, P00, C01, C11


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def scat_cov_axi(
    Ilm_in: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    normalisation: jnp.ndarray = None,
    filters: Tuple[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    if reality:
        Ilm = sphlib.make_flm_full(Ilm_in, L)
    else:
        Ilm = Ilm_in

    ######## Mean and variance of the data
    # mean = < I >_omega = I_00/2sqrt(pi) # [Nimg]
    mean = jnp.abs(
        Ilm[0, L - 1] / (2 * jnp.sqrt(jnp.pi))
    )  # TODO: VERY slight difference here jnp.real
    # var = Var(I) = <|I_lm|^2>_lm  # [Nimg]
    var = jnp.mean(jnp.abs(Ilm[1:]) ** 2)  # TODO: 4pi normalisations, complex type

    ######## COMPUTE C01 AND C11

    ######## Compute M_lm = |I * Psi|_lm
    ### Wavelet transform
    # W_lm = (I * Psi)_lm = I_lm * Psi_l0  # [Nimg, Q, Nalm]
    ### Go to map space and take the module
    # M = |SHT_inverse(W_lm)|  # [Nimg, Q, Npix]
    multires = False if sampling.lower() == "healpix" else multiresolution

    # TODO: some normalisation
    W, _ = s2wav.flm_to_analysis(
        Ilm,
        L,
        N,
        J_min,
        sampling=sampling,
        nside=nside,
        reality=reality,
        multiresolution=multires,
        filters=filters,
    )

    J = s2wav.utils.shapes.j_max(L)

    ### Go back to alm space
    # M_lm = SHT_forward(M)  # [Nimg, Q, Nalm]

    M_lm = []
    for j in range(J_min, J + 1):
        Lj, _, _ = s2wav.utils.shapes.LN_j(L, j, N, multiresolution=multires)
        val = s2fft.forward_jax(
            jnp.abs(W[j - J_min][0]),
            Lj,
            0,
            sampling=sampling,
            nside=nside,
            reality=reality,
        )
        M_lm.append(val)

    ######## COMPUTE S1 AND P00
    # S1 =  <M>_omega = M_00/2sqrt(pi)   # [Nimg, Q]
    S1 = []
    for j in range(J_min, J + 1):
        Lj, _, _ = s2wav.utils.shapes.LN_j(L, j, N, multiresolution=multiresolution)
        val = jnp.real(M_lm[j - J_min][0, Lj - 1]) / (2 * jnp.sqrt(jnp.pi))
        if normalisation is not None:
            val /= jnp.sqrt(normalisation[j - J_min])
        S1.append(val)

    # P00 = <M^2>_omega =  <WW*>_omega = <W_lm W_lm*>_lm = < |M_lm|^2 >_lm   # [Nimg, Q]
    P00 = []
    for j in range(J_min, J + 1):
        Lj, _, _ = s2wav.utils.shapes.LN_j(L, j, N, multiresolution=multiresolution)
        val = jnp.mean(
            jnp.abs(M_lm[j - J_min]) ** 2
        )  # TODO: This IS NOT the correct mean function
        val *= Lj / L
        if normalisation is not None:
            val /= normalisation[j - J_min]
        P00.append(val)

    ### Get |Psi_lm|^2 TODO: this can be done a priori
    # |Psi_lm(q)|^2 =  # [Q, Nalm]
    wav_lm_square = []
    for j in range(J_min, J + 1):
        val = jnp.abs(filters[0][j - J_min][:, L - 1]) ** 2  # TODO: THIS IS WRONG
        wav_lm_square.append(val)

    ### Compute C01 and C11
    # C01 = <I_lm M_lm(q1)* |Psi_lm(q2)|^2 >_lm  # [Nimg, Q1, Q2]
    # C11 = <M_lm(q1) M_lm(q2)* |Psi_lm(q3)|^2 >_lm  # [Nimg, Q1, Q2, Q3]

    C01 = []
    for j2 in range(J_min + 1, J + 1):
        L2j, _, _ = s2wav.utils.shapes.LN_j(L, j2, N, multiresolution=multiresolution)
        M_lm_q2 = M_lm[j2 - J_min]
        for j1 in range(J_min, j2):
            wav_lm_square_current = wav_lm_square[j1 - J_min]
            Lj, _, L0j = s2wav.utils.shapes.LN_j(
                L, j1, N, multiresolution=multiresolution
            )
            val = jnp.mean(
                jnp.einsum(
                    "lm,l->lm",
                    Ilm[L0j:Lj, L - Lj : L - 1 + Lj]
                    * jnp.conj(M_lm_q2[L0j:Lj, L2j - Lj : L2j - 1 + Lj]),
                    wav_lm_square_current[L0j:Lj],
                    optimize=True,
                )
            )  # TODO: Not the same mean function.
            if normalisation is not None:
                val /= jnp.sqrt(normalisation[j2 - J_min] * normalisation[j1 - J_min])
            C01.append(val)

    C11 = []
    for j3 in range(J_min + 2, J + 1):
        L3j, _, _ = s2wav.utils.shapes.LN_j(L, j3, N, multiresolution=multiresolution)
        M_lm_q1 = M_lm[j3 - J_min]
        for j2 in range(J_min + 1, j3):
            L2j, _, _ = s2wav.utils.shapes.LN_j(
                L, j2, N, multiresolution=multiresolution
            )
            M_lm_q2 = M_lm[j2 - J_min]
            for j1 in range(J_min, j2):
                Lj, _, L0j = s2wav.utils.shapes.LN_j(
                    L, j1, N, multiresolution=multiresolution
                )
                wav_lm_square_current = wav_lm_square[j1 - J_min]
                val = jnp.mean(
                    jnp.einsum(
                        "lm,l->lm",
                        M_lm_q1[L0j:Lj, L3j - Lj : L3j - 1 + Lj]
                        * jnp.conj(M_lm_q2[L0j:Lj, L2j - Lj : L2j - 1 + Lj]),
                        wav_lm_square_current[L0j:Lj],
                        optimize=True,
                    )
                )  # TODO: Not the same mean function.
                if normalisation is not None:
                    val /= jnp.sqrt(
                        normalisation[j3 - J_min] * normalisation[j2 - J_min]
                    )
                C11.append(val)

    return mean, var, jnp.array(S1), jnp.array(P00), jnp.array(C01), jnp.array(C11)


def quadrature(
    L: int,
    J_min: int = 0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = True,
):
    J = s2wav.utils.shapes.j_max(L)
    quads = []
    for j in range(J_min, J + 1):
        Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, 2.0, multiresolution)
        quads.append(s2fft.utils.quadrature_jax.quad_weights(Lj, sampling, nside))
    return quads


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from jax.lib import xla_bridge

    print(xla_bridge.get_backend().platform)

    L = 16
    N = 3
    J_min = 0
    sampling = "mw"
    reality = True
    multiresolution = True

    # Generate precomputed values
    filters = s2wav.filter_factory.filters.filters_directional_vectorised(L, N, J_min)
    weights = quadrature(L)
    precomps = s2wav.transforms.jax_wavelets.generate_wigner_precomputes(L, N, J_min, 2.0, sampling, None, False, reality, multiresolution)

    Ilm = np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1)

    mean, var, S1, P00, C01, C11 = scat_cov_dir(
        Ilm[:, L - 1 :],
        L,
        N,
        J_min,
        sampling,
        None,
        reality,
        multiresolution,
        filters=filters[0],
        quads=weights,
        precomps=precomps,
    )
    print(f"S1 shape = {S1.shape}")
    print(f"P00 shape = {P00.shape}")
    print(f"C01 shape = {C01.shape}")
    print(f"C11 shape = {C11.shape}")
