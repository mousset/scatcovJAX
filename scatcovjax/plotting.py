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
    fig.colorbar(im, ax=ax, orientation='horizontal')
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


def plot_alm(alm, s=10, vmin=None, vmax=None, lmin=None, lmax=None, mmin=None, mmax=None,
             cmap='viridis', figsize=(12, 6), fontsize=16, real_part_only=False):
    """
    Plot the alm in the (l, m) plane. By default, it plots real part, imaginary part and amplitude.
    TODO: Update because alm are 2D arrays now
    Parameters
    ----------
    alm: tensor
        alm hpx coefficients.
    s: int
        Size of the points.
    real_part_only: bool
        if True, only plot the real part.

    Returns
    -------

    """
    # ell, em = self.l_hpx, self.m_hpx

    if real_part_only:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im0 = ax.scatter(ell, em, c=np.real(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=ax)
        ax.set_xlabel(r'$\ell$', fontsize=fontsize)
        ax.set_ylabel('m', fontsize=fontsize)
        ax.axis('scaled')
        ax.set_xlim(lmin, lmax)
        ax.set_ylim(mmin, mmax)
        ax.set_title('Real part', fontsize=fontsize)
        fig.tight_layout()
    else:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        axs = np.ravel(axs)
        im0 = axs[0].scatter(ell, em, c=np.real(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=axs[0])
        axs[0].set_xlabel(r'$\ell$', fontsize=fontsize)
        axs[0].set_ylabel('m', fontsize=fontsize)
        axs[0].set_xlim(lmin, lmax)
        axs[0].set_ylim(mmin, mmax)
        axs[0].set_title('Real part', fontsize=fontsize)

        im1 = axs[1].scatter(ell, em, c=np.imag(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=axs[1])
        axs[1].set_xlabel(r'$\ell$', fontsize=fontsize)
        axs[1].set_ylabel('m', fontsize=fontsize)
        axs[1].set_xlim(lmin, lmax)
        axs[1].set_ylim(mmin, mmax)
        axs[1].set_title('Im part', fontsize=fontsize)

        im2 = axs[2].scatter(ell, em, c=np.abs(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im2, ax=axs[2])
        axs[2].set_xlabel(r'$\ell$', fontsize=fontsize)
        axs[2].set_ylabel('m', fontsize=fontsize)
        axs[2].set_xlim(lmin, lmax)
        axs[2].set_ylim(mmin, mmax)
        axs[2].set_title('Amplitude', fontsize=fontsize)
        fig.tight_layout()
    return fig