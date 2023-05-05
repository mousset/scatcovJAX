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


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def scat_cov_dir(
    Ilm_in: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    normalisation: jnp.ndarray = None,
    filters: jnp.ndarray = None,  # [J, L, 2L-1]
    quads: List[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> List[jnp.ndarray]:

    J_max = s2wav.utils.shapes.j_max(L)  # Maximum scale
    J = J_max - J_min + 1  # Number of scales
    print(f'{J=}')

    if quads is None:
        quads = []
        for j in range(J_min, J_max + 1):
            Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, 2.0, multiresolution)
            quads.append(s2fft.utils.quadrature_jax.quad_weights(Lj, sampling, nside))

    if precomps == None:
        precomps = s2wav.transforms.jax_wavelets.generate_wigner_precomputes(
            L, N, J_min, 2.0, sampling, nside, False, reality, multiresolution
        )  # [J][J][...]
    print('precomps', len(precomps), len(precomps[0]), precomps[0][0].shape, precomps[0][-1].shape)

    if reality:
        Ilm = sphlib.make_flm_full(Ilm_in, L)  # [L, 2L-1]
    else:
        Ilm = Ilm_in  # [L, 2L-1]

    # Compute mean and variance
    mean = jnp.abs(Ilm[0, L - 1] / (2 * jnp.sqrt(jnp.pi)))  # Take the I_00
    # var = jnp.mean(jnp.abs(Ilm[1:]) ** 2)
    var = jnp.sum(jnp.abs(Ilm[1:]) ** 2) / (4 * np.pi)  # Sum all except the (l=0, m=0) term

    ### Perform first (full-scale) wavelet transform W_jgamma = I * Psi_jgamma
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
    )  # [J][Norient, Ntheta, Nphi]=[J][2N-1, L, 2L-1]
    print('W', len(W))

    # Initialize S1, P00, Njjprime: will be list of len J
    S1 = []
    P00 = []
    Njjprime = []
    for j in range(J_min, J_max + 1):
        print(f'\n {j=}')
        # Subsampling: the resolution in the plane (l, m) is adapted at each scale j
        Lj = s2wav.utils.shapes.wav_j_bandlimit(L, j, 2.0, multiresolution)  # Band limit at resolution j
        Jmaxj = s2wav.utils.shapes.j_max(Lj)
        Jj = Jmaxj - J_min + 1  # Number of scales
        print(f'{Lj=}')
        print(f'{Jmaxj=}')
        print(f'{Jj=}')
        Njjprime_for_j = []

        def harmonic_step_for_j(n, args):
            """Compute M_lm = |W|_lm for one orientation [Lj, Mj] = [Lj, 2Lj-1]"""
            M_lm = args
            M_lm = M_lm.at[n].add(
                s2fft.forward_jax(
                    jnp.abs(W[j - J_min][n]),
                    Lj,
                    0,
                    sampling=sampling,
                    nside=nside,
                    reality=reality,
                )
            )
            return M_lm

        ### Compute M_lm for all orientations
        # Initialization
        M_lm = jnp.zeros((2 * N - 1, Lj, 2 * Lj - 1), dtype=jnp.complex128)  # [Norient, Lj, Mj]
        # Loop on orientations
        M_lm = lax.fori_loop(0, 2 * N - 1, harmonic_step_for_j, M_lm)  # [Norient, Lj, Mj]

        ### Compute S1
        # Take the value at (l=0, m=0) which corresponds to indices (0, Lj-1)
        val = M_lm[:, 0, Lj - 1] / (2 * jnp.sqrt(jnp.pi))  # [Norient]
        S1.append(val)  # [J][Norient]

        ### Compute P00
        # Average over Thetas, Phis
        val = jnp.mean(jnp.abs(W[j - J_min]) ** 2, axis=(-1, -2))  # [Norient]
        #val = jnp.sum(jnp.abs(W[j - J_min]) ** 2, axis=(-1, -2)) / (4 * np.pi)
        val *= Lj / L  # Normalization so that for each scale j, we divide by L, not Lj
        P00.append(val)  # [J][Norient]

        ### Compute Njjprime
        # TODO: This loop will increase compile time.
        #print('Jj', j-J_min+1)
        for n in range(2 * N - 1):  # Loop on orientations
            # Wavelet transform of Mlm
            val = s2wav.flm_to_analysis(
                M_lm[n],
                Lj,
                N,
                J_min,
                J_max=j-1,  # TODO ?? Pourquoi pas Jmaxj ?
                sampling=sampling,
                nside=nside,
                reality=reality,
                multiresolution=multiresolution,
                filters=filters[: j-J_min+1, :Lj, L - Lj: L - 1 + Lj],  # TODO??  Pourquoi pas [J_min: Jmaxj + 1]
                precomps=precomps[:j]  # TODO ?? Pourquoi pas [J_min: Jmaxj + 1]
            )  # [Jj][Norient, Nthetaj, Nphij]
            print('val', len(val), val[j].shape)
            Njjprime_for_j.append(val)  # [Norient][Jj][Norient, Nthetaj, Nphij]
        print('Njjprime_for_j', len(Njjprime_for_j), len(Njjprime_for_j[-1]), Njjprime_for_j[0][0].shape)
        Njjprime.append(Njjprime_for_j)  # [J][Norient][Jj][Norient, Nthetaj, Nphij]

    ### Reorder and flatten Njjprime, convert to JAX arrays for C01/C11
    Njjprime_flat = []
    for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1 TODO ?? Pourquoi pas Jmax+1? parce que dans le boucle suivante si j1=Jmax il ne se passe rien
        Njjprime_flat_for_j2 = []
        for j2 in range(j1 + 1, J_max + 1):  # j1+1 <= j2 <= J_max
            Njjprime_flat_for_n2 = []
            for n2 in range(2 * N - 1):
                Njjprime_flat_for_n2.append(Njjprime[j2 - J_min][n2][j1 - J_min][:, :, :])  # [Norient2][Norient1, Nthetaj1, Nphij1]
            Njjprime_flat_for_j2.append(Njjprime_flat_for_n2)  # [Jj2][Norient2][Norient1, Nthetaj1, Nphij1]
        Njjprime_flat.append(jnp.array(Njjprime_flat_for_j2))  # [J-1][Jj2, Norient2, Norient1, Nthetaj1, Nphij1]
    print('Njjprime_flat', len(Njjprime_flat), Njjprime_flat[0].shape, Njjprime_flat[-1].shape)

    ### Compute C01 and C11
    # Indexing: a/b = j3/j2, j/k = n3/n2, n = n1, theta = t, phi = p
    C01 = []
    C11 = []
    for j1 in range(J_min, J_max):  # J_min <= j1 <= J_max-1
        # Compute C01
        val = jnp.einsum(
            "ajntp,ntp->ajntp", jnp.conj(Njjprime_flat[j1 - J_min]), W[j1 - J_min], optimize=True
        )
        val = jnp.einsum("ajntp,t->ajn", val, quads[j1 - J_min], optimize=True)
        C01.append(val)  # [J1-1][J3, Norient3, Norient1]

        # Compute C11
        val = Njjprime_flat[j1 - J_min]
        val = jnp.einsum("ajntp,bkntp->abjkntp", val, jnp.conj(val), optimize=True)
        val = jnp.einsum("abjkntp,t->abjkn", val, quads[j1 - J_min], optimize=True)
        C11.append(val)  # [J1-1][J3, J2, Norient3, Norient2, Norient1]

    ### Make jnp array instead of list
    # S1 = jnp.concatenate(S1)  # [NS1] = [Norient x J]
    # P00 = jnp.concatenate(P00)  # [NP00] = [Norient x J]
    # C01 = jnp.concatenate(C01, axis=None)  # [NC01]
    # C11 = jnp.concatenate(C11, axis=None)  # [NC11]
    # print('S1, P00, C01, C11', S1.shape, P00.shape, C01.shape, C11.shape)
    #
    # ### Normalize S1 and P00
    # if normalisation is not None:
    #     S1 /= jnp.sqrt(normalisation)
    #     P00 /= normalisation

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
