from jax import jit, config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple

import s2wav
import s2fft

# __all__ = ['Scattering_sph']

"""
This library is made to
- compute the scattering covariance coefficients (axisym or directional)
- plot the coefficients

!!! TO DO:
- subsampling or not ? (yes)
- normalisation of the coeff by P00
- functions to plot the coeffs
- do as JM, return a class with the 4 coeffs ? 
"""


# class Scattering_sph(Spherical_Compute):

#     def __init__(self, nside, J, N, Psi_lm):
#         """

#         Parameters
#         ----------
#         nside: int
#         J: int
#             Number of j scales.
#         N: int
#             Number of orientations.
#             N=1 for Axisym and must be odd for directional.
#         Psi_lm: tensor array
#             alm of the wavelet. [Q, Nalm]
#         """
#         super().__init__(nside)
#         self.J, self.N = J, N
#         self.Q = J
#         self.Psi_lm = Psi_lm

#     def q2j(self, q):
#         j = self.J - q - 1
#         return j

#     def j2q(self, j):
#         q =  self.Q - j - 1
#         return q

#     def scat_cov_axi(self, Ilm, direct=True):

#         ######## Mean and variance of the data
#         # mean = < I >_omega = I_00/2sqrt(pi) # [Nimg]
#         # var = Var(I) = <|I_lm|^2>_lm  # [Nimg]

#         ######## COMPUTE C01 AND C11
#         if direct:
#             ######## Compute M_lm = |I * Psi|_lm
#             ### Wavelet transform
#             # W_lm = (I * Psi)_lm = I_lm * Psi_l0  # [Nimg, Q, Nalm]
#             ### Go to map space and take the module
#             # M = |SHT_inverse(W_lm)|  # [Nimg, Q, Npix]
#             ### Go back to alm space
#             # M_lm = SHT_forward(M)  # [Nimg, Q, Nalm]

#             ######## COMPUTE S1 AND P00
#             # S1 =  <M>_omega = M_00/2sqrt(pi)   # [Nimg, Q]
#             # P00 = <M^2>_omega =  <WW*>_omega = <W_lm W_lm*>_lm = < |M_lm|^2 >_lm   # [Nimg, Q]

#             ### Get |Psi_lm|^2
#             # |Psi_lm(q)|^2 =  # [Q, Nalm]

#             ### Compute C01 and C11
#             # C01 = <I_lm M_lm(q1)* |Psi_lm(q2)|^2 >_lm  # [Nimg, Q1, Q2]
#             # C11 = <M_lm(q1) M_lm(q2)* |Psi_lm(q3)|^2 >_lm  # [Nimg, Q1, Q2, Q3]

#         else:  #### Multiscale approach
#             ### Initialize the coeffs as empty array
#             # S1 = # [Nimg, Q]
#             # P00 = # [Nimg, Q]
#             # C01 = # [Nimg, Q1, Q2]
#             # C11 = # [Nimg, Q1, Q2, Q3]
#             M_dic = {}
#             for j3 in range(0, self.J): # From small to large scales
#                 q3 = self.j2q(j3)
#                 ######## Compute M_lm(q3) = |I * Psi(q3)|_lm
#                 ### Wavelet transform
#                 # W_lm(q3) = (I * Psi(q3))_lm = I_lm * Psi_l0(q3)  # [Nimg, Nalm(q3)]
#                 ### Go to map space and take the module
#                 # M(q3) = |SHT_inverse(W_lm(q3))|  # [Nimg, Npix(q3)]
#                 ### Go back to alm space
#                 # M_lm(q3) = SHT_forward(M(q3))  # [Nimg, Nalm(q3)]
#                 ### Store M_lm(q) in M_dic
#                 # M_dic[q3] = M_lm(q3)

#                 ######## COMPUTE S1 AND P00
#                 # S1[:, q3] =  <M(q3)>_omega = M_00(q3)/2sqrt(pi)   # [Nimg]
#                 # P00[:, q3] = <M^2(q3)>_omega =  <WW*>_omega = <W_lm W_lm*>_lm = < |M_lm|^2 >_lm   # [Nimg]

#                 # Get |Psi_3lm|^2
#                 # |Psi_lm(q3)|^2 =  # [Nalm(q3)]

#                 ### C01
#                 for j2 in range(0, j3):  # j2 <= j3
#                     q2 = self.j2q(j2)
#                     # Use Parseval at q3 resolution to simplify the calculation
#                     # C01[:, q2, q3] = < I_lm M_lm(q2)* |Psi_lm(q3)|^2 >_lm  # [Nimg]

#                     ### C11
#                     for j1 in range(0, j2):  # j1 <= j2
#                         q1 = self.j2q(j1)
#                         # Use Parseval at q3 resolution to simplify the calculation
#                         # C11[:, q1, q2, q3] = <M_lm(q1) M_lm(q2)* |Psi_lm(q3)|^2 >_lm  # [Nimg]

#                 ######## M_dic and I_lm are put at j3+1 resolution

#         ### Normalize C01 and C11
#         # C01 /= (P00_norm[:, :, None] *
#         #         P00_norm[:, None, :]) ** 0.5  # [Nimg, Q1, Q2]
#         # C11 /= (P00_norm[:, :, None, None] *
#         #         P00_norm[:, None, :, None]) ** 0.5  # [Nimg, Q1, Q2, Q3]

#         return mean, var, S1, P00, C01, C11

#     def scat_cov_dir(self, Ilm, direct=True):

#         ######## Mean and variance of the data
#         # mean = < I >_omega = I_00/2sqrt(pi) # [Nimg]
#         # var = Var(I) = <|I_lm|^2>_lm  # [Nimg]

#         ######## COMPUTE C01 AND C11
#         if direct:
#             ######## Compute M_lm = |I * Psi|_lm
#             ### Wavelet transform: directional convolution I_lm, Psi_lm => W
#             # W = (I * Psi) # [Nimg, Q, N, Npix]
#             ### Take the module
#             # M = |W|  # [Nimg, Q, N, Npix]
#             ### Go back to alm space
#             # M_lm = SHT_forward(M)  # [Nimg, Q, N, Nalm]

#             ######## COMPUTE S1 AND P00
#             # S1 =  <M>_omega = M_00/2sqrt(pi)   # [Nimg, Q]
#             # P00 = <M^2>_omega =  <WW*>_omega = <W_lm W_lm*>_lm = < |M_lm|^2 >_lm   # [Nimg, Q]

#             ######## Compute C01 and C11
#             # C01 = PS[I*Psi(j3), M(j2)*Psi(j3)] # [Nimg, Q1, Q2, N1, N2]
#             # C11 = PS[M(j1)*Psi(j3), M(j2)*Psi(j3)] # [Nimg, Q1, Q2, Q3, N1, N2, N3]


#         else:  #### Multiscale approach
#             ### Initialize coeffs as empty array
#             # S1 = # [Nimg, Q]
#             # P00 = # [Nimg, Q]
#             # C01 = # [Nimg, Q1, Q2, N1, N2]
#             # C11 = # [Nimg, Q1, Q2, Q3, N1, N2, N3]
#             M_dic = {}
#             for j3 in range(0, self.J):  # From small to large scales
#                 q3 = self.j2q(j3)
#                 ######## Compute M_lm(q3) = |I * Psi(q3)|_lm
#                 ### Wavelet transform: directional convolution I_lm, Psi_lm => W
#                 # W(q3) = (I * Psi(q3)) # [Nimg, N, Npix(q3)]
#                 ### Go to map space and take the module
#                 # M(q3) = |W(q3)|  # [Nimg, Npix(q3)]
#                 ### Go back to alm space
#                 # M_lm(q3) = SHT_forward(M(q3))  # [Nimg, Nalm(q3)]
#                 ### Store M_lm(q) in M_dic
#                 # M_dic[q3] = M_lm(q3)

#                 ######## COMPUTE S1 AND P00
#                 # S1[:, q3] =  <M(q3)>_omega = M_00(q3)/2sqrt(pi)   # [Nimg]
#                 # P00[:, q3] = <M^2(q3)>_omega =  <WW*>_omega = <W_lm W_lm*>_lm = < |M_lm|^2 >_lm   # [Nimg]

#                 ##### C01
#                 for j2 in range(0, j3):  # j2 <= j3
#                     q2 = self.j2q(j2)
#                     ### Compute N_lm(q2, q3) = |M(q2) * Psi(q3)_lm

#                     ### Store N_lm(q2, q3)

#                     # C01[:, q2, q3] =  # [Nimg]

#                     ##### C11
#                     for j1 in range(0, j2):  # j1 <= j2
#                         q1 = self.j2q(j1)
#                         # C11[:, q1, q2, q3] =  # [Nimg]

#                 ######## M_dic and I_lm are put at j3+1 resolution

#         ### Normalize C01 and C11
#         # C01 /= (P00_norm[:, :, None, :, None] *
#         #         P00_norm[:, None, :, None, :]) ** 0.5  # [Nimg, Q1, Q2, N1, N2]
#         # C11 /= (P00_norm[:, :, None, None, :, None, None] *
#         #         P00_norm[:, None, :, None, None, :, None]) ** 0.5  # [Nimg, Q1, Q2, Q3, N1, N2, N3]

#         return mean, var, S1, P00, C01, C11


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
    flat_covariances: bool = True,
    normalisation: jnp.ndarray = None,
    filters: Tuple[jnp.ndarray] = None,
) -> List[jnp.ndarray]:

    if reality:
        # Create and store signs 
        msigns = (-1)**jnp.arange(1,L)

        # Reflect and apply hermitian symmetry
        Ilm = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
        Ilm = Ilm.at[:,L-1:].set(Ilm_in)
        Ilm = Ilm.at[:, : L - 1].set(jnp.flip(jnp.conj(Ilm[:, L:])*msigns, axis=-1))

    else:
        Ilm = Ilm_in

    ######## Mean and variance of the data
    # mean = < I >_omega = I_00/2sqrt(pi) # [Nimg]
    mean = jnp.abs(Ilm[0, L - 1] / (2 * jnp.sqrt(jnp.pi)))  #TODO: VERY slight difference here jnp.real 
    # var = Var(I) = <|I_lm|^2>_lm  # [Nimg]
    var = jnp.mean(jnp.abs(Ilm[1:])**2) #TODO: 4pi normalisations, complex type

    ######## COMPUTE C01 AND C11

    ######## Compute M_lm = |I * Psi|_lm
    ### Wavelet transform
    # W_lm = (I * Psi)_lm = I_lm * Psi_l0  # [Nimg, Q, Nalm]
    ### Go to map space and take the module
    # M = |SHT_inverse(W_lm)|  # [Nimg, Q, Npix]
    if sampling == 'healpix':
        multires = False
    else:
        multires = multiresolution

    #TODO: some normalisation 
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
        Lj, _, _ = s2wav.utils.shapes.LN_j(
            L, j, N, multiresolution=multires
        )
        val = s2fft.forward_jax(
            jnp.abs(W[j - J_min][0]), Lj, 0, sampling=sampling, nside=nside, reality=reality,
        )
        M_lm.append(val)

    ######## COMPUTE S1 AND P00
    # S1 =  <M>_omega = M_00/2sqrt(pi)   # [Nimg, Q]
    S1 = []
    for j in range(J_min, J + 1):
        Lj, _, _ = s2wav.utils.shapes.LN_j(
            L, j, N, multiresolution=multiresolution
        )
        val = jnp.real(M_lm[j - J_min][0, Lj - 1]) / (2 * jnp.sqrt(jnp.pi))
        if normalisation is not None:
            val /= jnp.sqrt(normalisation[j-J_min])
        S1.append(val)

    # P00 = <M^2>_omega =  <WW*>_omega = <W_lm W_lm*>_lm = < |M_lm|^2 >_lm   # [Nimg, Q]
    P00 = []
    for j in range(J_min, J + 1):
        Lj, _, _ = s2wav.utils.shapes.LN_j(
            L, j, N, multiresolution=multiresolution
        )
        val = jnp.mean(jnp.abs(M_lm[j - J_min])**2) #TODO: This IS NOT the correct mean function
        val *= Lj / L
        if normalisation is not None:
            val /= normalisation[j-J_min]
        P00.append(val)

    ### Get |Psi_lm|^2 TODO: this can be done a priori
    # |Psi_lm(q)|^2 =  # [Q, Nalm]
    wav_lm_square = []
    for j in range(J_min, J + 1):
        val = jnp.abs(filters[0][j - J_min][:,L-1])**2  #TODO: THIS IS WRONG 
        wav_lm_square.append(val)

    ### Compute C01 and C11
    # C01 = <I_lm M_lm(q1)* |Psi_lm(q2)|^2 >_lm  # [Nimg, Q1, Q2]
    # C11 = <M_lm(q1) M_lm(q2)* |Psi_lm(q3)|^2 >_lm  # [Nimg, Q1, Q2, Q3]

    C01 = []
    for j2 in range(J_min + 1, J + 1):
        L2j, _, _ = s2wav.utils.shapes.LN_j(
            L, j2, N, multiresolution=multiresolution
        )
        M_lm_q2 = M_lm[j2 - J_min]
        C01_q2 = []
        for j1 in range(J_min, j2):
            wav_lm_square_current = wav_lm_square[j1 - J_min]
            Lj, _, L0j = s2wav.utils.shapes.LN_j(
                L, j1, N, multiresolution=multiresolution
            )
            val = jnp.mean(
                jnp.einsum("lm,l->lm",
                Ilm[L0j:Lj, L - Lj : L - 1 + Lj]
                * jnp.conj(M_lm_q2[L0j:Lj, L2j - Lj : L2j - 1 + Lj]),
                wav_lm_square_current[L0j:Lj], optimize=True)
            ) #TODO: Not the same mean function.
            if normalisation is not None:
                val /= jnp.sqrt(normalisation[j2 - J_min] * normalisation[j1 - J_min])
            C01_q2.append(val)
        C01.append(C01_q2)

    C11 = []
    for j3 in range(J_min + 2, J + 1):
        L3j, _, _ = s2wav.utils.shapes.LN_j(
            L, j3, N, multiresolution=multiresolution
        )
        M_lm_q1 = M_lm[j3 - J_min]
        C11_q1 = []
        for j2 in range(J_min + 1, j3):
            L2j, _, _ = s2wav.utils.shapes.LN_j(
                L, j2, N, multiresolution=multiresolution
            )
            M_lm_q2 = M_lm[j2 - J_min]
            C11_q2 = []
            for j1 in range(J_min, j2):
                Lj, _, L0j = s2wav.utils.shapes.LN_j(
                    L, j1, N, multiresolution=multiresolution
                )
                wav_lm_square_current = wav_lm_square[j1 - J_min]
                val = jnp.mean(jnp.einsum("lm,l->lm",
                    M_lm_q1[L0j:Lj, L3j - Lj : L3j - 1 + Lj]
                    * jnp.conj(M_lm_q2[L0j:Lj, L2j - Lj : L2j - 1 + Lj]),
                    wav_lm_square_current[L0j:Lj], optimize=True)
                ) #TODO: Not the same mean function.
                if normalisation is not None:
                    val /= jnp.sqrt(normalisation[j3 - J_min] * normalisation[j2 - J_min])
                C11_q2.append(val)
            C11_q1.append(C11_q2)
        C11.append(C11_q1)

    if flat_covariances:
        # Flatten lists and convert to arrays
        C01 = jnp.stack([item for sublist in C01 for item in sublist])
        C11 = jnp.stack([item for sublist in C11 for subsublist in sublist for item in subsublist])

    return mean, var, jnp.array(S1), jnp.array(P00), C01, C11

