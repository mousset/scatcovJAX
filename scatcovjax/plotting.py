import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import s2fft


def notebook_plot_format():
    plt.rc('font', size=16)  # controls default text sizes
    plt.rc('axes', titlesize=16)  # fontsize of the axes title
    plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('figure', titlesize=16)  # fontsize of the figure title
    return


def plot_map_MW_Mollweide(map_MW, figsize=(12, 8), fontsize=16, vmin=None, vmax=None,
                          central_longitude=0, title='Map - Real part',
                          fig=None, ax=None):

    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=central_longitude))
    im = ax.imshow(np.real(map_MW), transform=ccrs.PlateCarree(),
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=fontsize)
    #fig.colorbar(im, ax=ax, orientation='horizontal')
    fig.tight_layout()
    return


def plot_map_MW_Orthographic(map_MW, figsize=(12, 8), fontsize=16, vmin=None, vmax=None,
                          central_longitude=0., central_latitude=0.0, title='Map - Real part',
                          fig=None, ax=None):

    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=central_longitude,
                                                   central_latitude=central_latitude,
                                                   globe=None))
    im = ax.imshow(np.real(map_MW), transform=ccrs.PlateCarree(),
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=fontsize)
    fig.colorbar(im, ax=ax, orientation='horizontal')
    return


def plot_sphere(map, L, sampling, isnotebook=True, cmap='viridis'):
    """
    Nice interactive plot from S2FFT.
    """
    s2fft.utils.plotting_functions.plot_sphere(map, L=L, sampling=sampling, isnotebook=isnotebook, cmap=cmap)
    return


def plot_filters(filters, real=True, m=None, figsize=(8, 6)):
    wlm, slm = filters  # Split scaling function and wavelets
    if real:
        wlm = np.real(wlm)
    else:
        wlm = np.imag(wlm)
    J = wlm.shape[0]  # Number of wavelets
    fig = plt.subplots(1, 1, figsize=figsize)
    plt.plot(slm, 'k', label='Scaling fct')
    for j in range(J):
        if m is None:  # Axisym filters
            plt.plot(wlm[j, :], label=f'Wavelet {j}')
        else:  # Directionnal filters
            plt.plot(wlm[j, :, m], label=f'Wavelet {j}')
    plt.xscale('log', base=2)
    plt.legend()
    return fig


def plot_alm(flm, vmin=None, vmax=None, lmin=None, lmax=None, mmin=None, mmax=None,
             cmap='viridis', figsize=(12, 6)):
    """
    Plot the flm in the (l, m) plane.
    flm: array
        2D array [L, 2L-1]
    """
    L = flm.shape[0]

    def for_all_plots(ax):
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$m$')
        ax.set_xlim(lmin, lmax)
        ax.set_ylim(mmin, mmax)
        ax.plot(np.arange(L + 1), np.arange(L + 1), 'white')
        ax.plot(np.arange(L + 1), -np.arange(L + 1), 'white')
        ax.grid()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    im0 = ax0.imshow(np.real(flm).T, origin='lower', extent=(0, L, -L, L), cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=ax0)
    ax0.set_title('Real part')
    for_all_plots(ax0)

    im1 = ax1.imshow(np.imag(flm).T, origin='lower', extent=(0, L, -L, L), cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Imag part')
    for_all_plots(ax1)

    fig.tight_layout()
    return fig


def plot_scatcov_coeffs(S1, P00, C01, C11, name=None, hold=True, color='blue', ls='-', marker=''):

    if name is None:
        name = ''

    if hold:
        plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.plot(np.real(S1), color=color, label=f'{name} ' + r'$S_1$', ls=ls, marker=marker)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(np.real(P00), color=color, label=f'{name} ' + r'$P_{00}$', ls=ls, marker=marker)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(np.real(C01), color=color, label=f'{name} ' + r'$C_{01}$', ls=ls, marker=marker)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(np.real(C11), color=color, label=f'{name} ' + r'$C_{11}$', ls=ls, marker=marker)
    plt.legend()

    return


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