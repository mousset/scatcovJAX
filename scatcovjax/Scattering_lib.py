from jax import jit, config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple

import scatcovjax.Sphere_lib as sphlib
import s2wav
import s2fft

# @partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
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
    filters: Tuple[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    if reality:
        sphlib.make_flm_full(Ilm_in, L)
    else:
        Ilm = Ilm_in
    
    ######## Mean and variance of the data
    # mean = < I >_omega = I_00/2sqrt(pi) # [Nimg]
    mean = jnp.abs(
        Ilm[0, L - 1] / (2 * jnp.sqrt(jnp.pi))
    )  
    # var = Var(I) = <|I_lm|^2>_lm  # [Nimg]
    var = jnp.mean(jnp.abs(Ilm[1:]) ** 2) 

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

    Njjprime = []
    S1 = []
    P00 = []
    for j in range(J_min, J + 1):
        Lj, Nj, _ = s2wav.utils.shapes.LN_j(L, j, N, multiresolution=multires)

        Njjprime_for_j = []
        for n in range(2*Nj-1):
            M_lm = s2fft.forward_jax(
                jnp.abs(W[j - J_min][n]),
                Lj,
                0,
                sampling=sampling,
                nside=nside,
                reality=reality,
            )

            # Compute S1
            val = jnp.real(M_lm[0, Lj - 1]) / (2 * jnp.sqrt(jnp.pi))
            if normalisation is not None:
                val /= jnp.sqrt(normalisation[j - J_min])
            S1.append(val)

            # Compute P00
            # The Lj/L term makes the mean comparable to the non-multiresolution mean.
            val = jnp.mean(jnp.abs(M_lm) ** 2) 
            val *= Lj / L    
            if normalisation is not None:
                val /= normalisation[j - J_min]
            P00.append(val)

            val, _ = s2wav.flm_to_analysis(
                M_lm,
                Lj,
                Nj,
                J_min,
                sampling=sampling,
                nside=nside,
                reality=reality,
                multiresolution=multires,
                filters=(filters[0][:j-J_min+1, :Lj, L-Lj:L-1+Lj], filters[1])
            )
            Njjprime_for_j.append(val)
        Njjprime.append(Njjprime_for_j)

    quads = []
    for j in range(J_min, J+1):
        Lj, _, _ = s2wav.utils.shapes.LN_j(L, j, N, multiresolution=multiresolution)
        quads.append(s2fft.utils.quadrature_jax.quad_weights(Lj, sampling, nside))

    C01 = []
    for j3 in range(J_min + 1, J + 1):
        _, N3j, _ = s2wav.utils.shapes.LN_j(L, j3, N, multiresolution=multiresolution)
        for n3 in range(2*N3j-1):
            for j2 in range(J_min, j3):
                L2j, N2j, _ = s2wav.utils.shapes.LN_j(L, j2, N, multiresolution=multiresolution)
                for n2 in range(2*N2j-1):
                    val = W[j2-J_min][n2]
                    val *= jnp.conj(Njjprime[j3-J_min][n3][j2-J_min][n2])
                    if sampling.lower() == "healpix":
                        val = jnp.sum(val * quads) #TODO: fix quad indexing for healpix multires.
                    else:
                        val = jnp.sum(jnp.einsum("tp,t->tp", val, quads[j2-J_min], optimize=True))
                    
                    if normalisation is not None:
                        val /= jnp.sqrt(normalisation[j3 - J_min] * normalisation[j2 - J_min])

                    C01.append(val) 
    
    C11 = []
    for j3 in range(J_min + 2, J + 1):
        _, N3j, _ = s2wav.utils.shapes.LN_j(L, j3, N, multiresolution=multiresolution)
        for n3 in range(2*N3j-1):
            for j2 in range(J_min + 1, j3):
                _, N2j, _ = s2wav.utils.shapes.LN_j(L, j2, N, multiresolution=multiresolution)

                for n2 in range(2*N2j-1):
                    for j1 in range(J_min, j2):
                        L1j, N1j, _ = s2wav.utils.shapes.LN_j(L, j1, N, multiresolution=multiresolution)
                        for n1 in range(2*N1j-1):
                            val = Njjprime[j2-J_min][n2][j1-J_min][n1]
                            val *= jnp.conj(Njjprime[j3-J_min][n3][j1-J_min][n1])
                            if sampling.lower() == "healpix":
                                val = jnp.sum(val * quads) #TODO: fix quad indexing for healpix multires.
                            else:  
                                val = jnp.sum(jnp.einsum("tp,t->tp", val, quads[j1-J_min], optimize=True))

                            if normalisation is not None:
                                val /= jnp.sqrt(normalisation[j3 - J_min] * normalisation[j2 - J_min])

                            C11.append(val) 
                    
    return mean, var, jnp.array(S1), jnp.array(P00), jnp.array(C01), jnp.array(C11)


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
        sphlib.make_flm_full(Ilm_in, L)
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


if __name__=="__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Check we're running on GPU
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)

    L = 32
    N = 3
    J_min = 0
    sampling="mw"
    reality=True
    multiresolution=True
    filters = s2wav.filter_factory.filters.filters_directional_vectorised(L, N, J_min)
    Ilm = np.random.randn(L, 2*L-1) + 1j*np.random.randn(L, 2*L-1)

    mean, var, S1, P00, C01, C11 = scat_cov_dir(Ilm[:,L-1:], L, N, J_min, sampling, None, reality, multiresolution, filters=filters)
