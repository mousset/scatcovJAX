# import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm

# import scatcovjax.Wavelets_lib as wlib

# """
# This library is made to
# - run a synthesis
# - control the synthesis output with plots

# !!! TO DO:
# - see how it is implemented in Foscat to define a loss and run a synthesis
# - see how to run an optimization in JAX
# """

# ############# PLOT TO CONTROL THE SYNTHESIS #################


# def plot_loss(loss, figsize=(6, 4), fontsize=16, color='b', title=''):
#     plt.figure(figsize=figsize)
#     plt.plot(loss, 'o', color=color)
#     plt.ylabel('Loss', fontsize=fontsize)
#     plt.yscale('log')
#     plt.xlabel('Iteration', fontsize=fontsize)
#     plt.title(title)
#     plt.grid()
#     return


# def plot_power_spectrum(target_Cl, ini_Cl, synthetic_Cl, J, filter_alm, m_MW, figsize=(10, 8), fontsize=16):
#     plt.figure(figsize=figsize)
#     plt.plot(target_Cl, 'b', label='Target')
#     plt.plot(ini_Cl, 'g', label='Initial')
#     plt.plot(synthetic_Cl, 'r', label='Synthesis')
#     colors = cm.get_cmap('viridis', J).colors
#     for j in range(J):
#         c = colors[j]
#         lmin_band, lmax_band = wlib.get_band_axisym_wavelet(filter_alm, m_MW, j_scale=j)
#         plt.axvline(lmin_band, color=c, ls='--')
#         plt.axvline(lmax_band, color=c, ls='--')
#         plt.axvspan(lmin_band, lmax_band, color=c, alpha=0.1)
#     plt.yscale('log')
#     plt.xlabel(r'$\ell$', fontsize=fontsize)
#     plt.ylabel(r'$C_\ell$', fontsize=fontsize)
#     plt.grid()
#     plt.legend(fontsize=fontsize)
#     return


# def plot_histo(target_hpx, synthetic_hpx, bins=50, range=(-5, 5), ymax=1300, fontsize=16):
#     plt.figure(figsize=(10, 8))
#     plt.hist(target_hpx.ravel(), bins=bins, range=range, color='b', alpha=0.3, label='Target')
#     plt.hist(synthetic_hpx.ravel(), bins=bins, range=range, color='r', alpha=0.3, label='Synthesis')
#     plt.legend(fontsize=fontsize)
#     plt.ylim(0, ymax)
#     plt.xlim(range)
#     plt.grid()
#     return
