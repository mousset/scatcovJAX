# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
# import pys2let as s2let

# """
# This library is made to
# - produce wavelets from S2LET (axisym or directional)
# - define the convolutions axisym and directional

# !!! TO DO:
# - see if we still use S2LET to produce wavelets
# - should we convert numpy array in JAX objects ?
# - how should we treat complex, split real and imag ?
# - convert the convolutions in JAX

# """


# def get_band_axisym_wavelet(filter_alm, m_MW, j_scale):
#     """
#     !!! Not very important (used in the plot of the filters)
#     Parameters
#     ----------
#     filter_alm: Tensor
#         Wavelet set [J, Nalm]
#     m_MW
#     j_scale: int

#     Returns
#     -------

#     """
#     theband = np.where(filter_alm.numpy()[j_scale, m_MW == 0] != 0. + 0.j)[0]
#     lmin_band = theband[0] - 1
#     lmax_band = theband[-1]
#     return lmin_band, lmax_band


# def get_J_number_of_wavelets(B, L, J_min):
#     """
#     !!! See if we still remove the last wavelet
#     Compute the number of wavelets.
#     Parameters
#     ----------
#     B: int
#         Wavelet parameter which determines the scale factor between consecutive wavelet scales.
#     L: int
#         Maximal l scale, l goes from 0 to L-1.
#     J_min: int
#         First wavelet scale to be used (0 is the scaling function).

#     Returns
#     -------
#     J: Number of wavelets.
#     """
#     # Maximal scale
#     # We do -1 because we decided to remove the last wavelet
#     J_max = s2let.pys2let_j_max(B, L, J_min) - 1
#     # Number of wavelets
#     J = J_max - J_min + 1
#     return J


# def make_axisym_wavelets(nside, B, J_min, l_MW, m_MW, doplot=False):
#     """
#     Make a set of axisymmetric filters.
#     All value at same m are equal to the one at m=0.
#     Parameters
#     ----------
#     nside: int
#         Nside parameter from Healpix, power of 2.
#     B: int
#         Wavelet parameter which determines the scale factor between consecutive wavelet scales.
#     J_min: int
#         First wavelet scale to be used (0 is the scaling function).
#     l_MW: Tensor
#         List of l, ordered in MW.
#     m_MW: Tensor
#         List of l, ordered in MW.
#     doplot: bool
#         If True, plot the filters.

#     Returns
#     -------
#     filter_alm_bis: Tensor [J, Nalm]
#     scaling_alm_bis: Tensor [Nalm]
#     """
#     lmax = 3 * nside - 1  # We use the Healpix default value
#     L = lmax + 1
#     Nalm = L ** 2
#     scal_l, wav_l = s2let.axisym_wav_l(B, L, J_min)
#     # We remove the last wavelet
#     wav_l = wav_l[:, :-1]
#     # Number of wavelets
#     J = get_J_number_of_wavelets(B, L, J_min)

#     filter_alm = np.zeros((J, Nalm), dtype='complex')  # [J, Nalm]
#     scaling_alm = np.zeros(Nalm, dtype='complex')  # [Nalm]

#     # Complete the alm plane with 0 where m is not 0.
#     filter_alm[:, m_MW == 0] = wav_l.T
#     scaling_alm[m_MW == 0] = scal_l.T

#     # Set all value m not zero to the value in m=0
#     # Trick used to replace the convolution by a simple product of arrays
#     filter_alm_bis = np.zeros_like(filter_alm, dtype='complex')
#     scaling_alm_bis = np.zeros_like(scaling_alm, dtype='complex')
#     for l in range(L):
#         filter_alm_bis[:, l == l_MW] = filter_alm[:, (l == l_MW) & (m_MW == 0)]
#         scaling_alm_bis[l == l_MW] = scaling_alm[(l == l_MW) & (m_MW == 0)]

#     if doplot:
#         plt.figure(figsize=(8, 6))
#         plt.plot(scal_l, 'k', label='Scaling fct')
#         colors = cm.get_cmap('viridis', J).colors
#         for j in range(J):
#             c = colors[j]
#             plt.plot(wav_l[:, j], color=c, label=f'Wav j={j}')
#             lmin_band, lmax_band = get_band_axisym_wavelet(filter_alm_bis, m_MW, j_scale=j)
#             plt.axvline(lmin_band, color=c, ls='--')
#             plt.axvline(lmax_band, color=c, ls='--')
#             plt.axvspan(lmin_band, lmax_band, color=c, alpha=0.2)
#         plt.xlabel(r'$\ell$', fontsize=16)
#         plt.xscale('log', base=2)
#         plt.legend(fontsize=16)

#     return scaling_alm_bis, filter_alm_bis


# def make_directional_wavelets(nside, B, N, J_min, m_MW, doplot=False):
#     """
#     Parameters
#     ----------
#     nside: int
#         Nside parameter from Healpix, power of 2.
#     B: int
#         Wavelet parameter which determines the scale factor between consecutive wavelet scales.
#     N: int
#         Number of orientations, N must be odd.
#     J_min: int
#         First wavelet scale to be used (0 is the scaling function).
#     m_MW:
#     doplot: bool

#     Returns
#     -------
#     filter_alm_bis: Tensor [J, Nalm]
#     scaling_alm_bis: Tensor [Nalm]
#     """
#     # Check that N is odd
#     # if N // 2 != 0:
#     #     raise ValueError('N must be odd.')

#     lmax = 3 * nside - 1  # We use the Healpix default value
#     L = lmax + 1
#     scal_ln, wav_ln = s2let.wavelet_tiling(B, L, N, J_min, spin=0, original_spin=0)
#     # We remove the last wavelet
#     wav_ln = wav_ln[:, :-1]  # [Nalm, J]
#     # Number of wavelets
#     J = get_J_number_of_wavelets(B, L, J_min)

#     wavelet_ln = wav_ln.T  # [J, Nalm]
#     scaling_ln = scal_ln.T  # [Nalm]

#     if doplot:
#         plt.figure(figsize=(8, 6))
#         plt.plot(np.real(scal_ln), 'k', label='Scaling fct')
#         for j in range(J):
#             plt.plot(np.real(wav_ln[m_MW == 0, j]), 'o', label=f'Wav j={j}', alpha=0.5)
#         plt.xscale('log', base=2)
#         plt.xlabel(r'$\ell$')
#         plt.title(f'Cut at m = 0')
#         plt.legend()

#     return scaling_ln, wavelet_ln

# ######################### CONVOLUTIONS ########################


# def make_axi_convolution(data_alm, wavelet_alm):
#     """
#     Axisym convolution in alm space.
#     data_alm and wavelet_alm must have the same shape.
#     !!! The wavelet alm must be modified such that
#     alm for m!=0 are set to the value at m=0.
#     Parameters
#     ----------
#     data_alm: tensor array
#         alm of the data [..., Nalm].
#     wavelet_alm: tensor array
#         alm of the wavelet [..., Nalm].
#     Returns
#     -------
#     conv_alm: The result of the convolution in alm space [..., Nalm].

#     """
#     # conv_alm = np.sqrt(4 * np.pi / (2 * l_MW + 1)) * data_alm * tnp.conj(filter_set_alm)
#     # !! Do not consider the 4pi/2l+1 factor in the Axisym convolution,
#     # ok as S2LET wavelets are not normalized
#     conv_alm = data_alm * np.conj(wavelet_alm)

#     return conv_alm


# def make_list_indices(L, N):
#     # We create the list of indices used to construct the wigner representation
#     list_lm, list_ln, list_n, list_l = [], [], [], []
#     for n in range(N):
#         for el in range(L):
#             for m in range(-el, el + 1):
#                 list_l.append(el)  # Index for l
#                 list_lm.append(ssht.elm2ind(el, m))  # Index for flm
#                 list_ln.append(ssht.elm2ind(el, n))  # Index for psi_ln
#                 list_n.append(n)  # Index for n
#     return [list_lm, list_ln, list_n, list_l]


# def make_directional_convolution(data_lm, wavelet_ln, J, N, L, list_idx):
#     """
#     Perform a directional convolution of a map by a wavelet set.
#     Parameters
#     ----------
#     data_lm: Tensor
#         alm of the data map to be convolved [..., Nalm].
#     wavelet_ln: Tensor
#         alm of the wavelets [J, Nalm]
#     J: int
#         Number of wavelet scales.
#     N: int
#         Number of orientations must be odd.
#     L: int
#         Maximal l scale, l goes from 0 to L-1.
#     list_idx: list
#         List of four lists with indices needed for directional convolution.
#     Returns
#     -------
#     wigner_coeffs_rho: the convolved map [..., J, N, Ntheta, Nphi]
#     """

#     Nalm = L ** 2
#     # Get the input shape (except Nalm)
#     shape_in = list(data_lm.shape[:-1])  # [A]

#     # Reshape data_lm in 2D
#     data_lm = tf.reshape(data_lm, shape=(-1, Nalm))  # [A, Nalm]

#     # The len of each list is NxNalm
#     list_lm, list_ln, list_n, list_l = list_idx

#     # We create the pre-factor array
#     prefactor = 8 * np.pi ** 2 / (2 * np.arange(L) + 1)

#     # Computation of the Wigner coefficients
#     wigner_coeffs_lmn = tf.gather(data_lm, axis=-1, indices=list_lm)[:, None, :] * \
#                         tf.gather(wavelet_ln, axis=-1, indices=list_ln) * \
#                         prefactor[list_l]  # [A, J, NxNalm]
#     wigner_coeffs_lmn = tf.reshape(wigner_coeffs_lmn,
#                                    shape=(shape_in + [J, N, Nalm]))  # [..., J, N, Nalm]
#     # Go to map space (Wigner transform)
#     wigner_coeffs_rho = wigner_layer(is_forward=False, sampling="mw", dtype=tf.complex64)(
#         wigner_coeffs_lmn)  # [..., J, Ntheta, Nphi, N]

#     # Reshape to [..., J, N, Ntheta, Nphi]
#     wigner_coeffs_rho = tnp.moveaxis(wigner_coeffs_rho, source=-1, destination=-3)

#     return wigner_coeffs_rho
