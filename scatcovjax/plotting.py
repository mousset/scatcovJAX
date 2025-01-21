import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import pynkowski as mf   # For Minkowski Functionals
import s2fft
#from mayavi import mlab
import s2wav
from s2fft.sampling import s2_samples as samples

import scatcovjax.Sphere_lib as sphlib


def notebook_plot_format():
    plt.rc('font', size=16)  # controls default text sizes
    plt.rc('axes', titlesize=16)  # fontsize of the axes title
    plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('figure', titlesize=16)  # fontsize of the figure title
    return


def get_hist_envelop(f, bins=100, range=(-10, 10), cumulative=False):
    """Return the fraction of pixels as a function of the pixel values.
    The histogram is normalized such that the sum is one."""
    hist, bins_edges = np.histogram(f, bins=bins, range=range, density=True)
    bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2  # Bin centers
    bw = bins_centers[1] - bins_centers[0]
    if cumulative:
        cumsum = np.cumsum(hist*bw)
        return bins_centers, cumsum
    else:
        return bins_centers, hist * bw


def make_minkowski(us, map_hpx):
    data = mf.Healpix(map_hpx, normalise=False, mask=None)  # Default parameters
    v0 = mf.V0(data, us)
    v1 = mf.V1(data, us)
    v2 = mf.V2(data, us)

    return v0, v1, v2


def plot_map_MW_Mollweide(map_MW, figsize=(12, 8), fontsize=16, vmin=None, vmax=None,
                          central_longitude=0, pole_latitude=90, pole_longitude=180, title='Map - Real part',
                          fig=None, ax=None, colorbar=False):
    rotated_pole = ccrs.RotatedPole(pole_latitude=pole_latitude, pole_longitude=pole_longitude)
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=central_longitude))
    im = ax.imshow(np.real(map_MW), transform=rotated_pole,
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=fontsize)
    # ccrs.PlateCarree()
    if colorbar:
        fig.colorbar(im, ax=ax, orientation='horizontal')
    fig.tight_layout()
    return fig


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


def plot_sphere(f: np.ndarray, L: int, sr: float, mx: float, mn: float):

    # Define meshgrid points on spherical surface
    phis = samples.phis_equiang(L, sampling="mw")
    thetas = samples.thetas(L, sampling="mw")

    # Fix continuity at boundaries for visualisation
    thetas[0] = 0
    phis[-1] = 2 * np.pi

    # Generate angular meshgrid
    phi, theta = np.meshgrid(phis, thetas)

    # Scaling to increase/decrease magnitude of coefficient for visualisation
    temp = (f - mn) / mx
    r = sr + temp

    # Convert angular meshgrid to cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # 3D render using mayavi package.
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(300, 300))
    mlab.clf()
    mlab.mesh(x, y, z, scalars=temp, colormap="viridis",vmax=1-mn/mx,vmin=0)
    mlab.show()


def plot_filters(filters, J_min, J_max, real=True, m=None, figsize=(8, 6)):
    """

    Parameters
    ----------
    filters
    real: bool
        If True, plot the real part of the filters.
    m
    figsize

    Returns
    -------

    """
    wlm, slm = filters  # Split scaling function and wavelets
    if real:
        wlm = np.real(wlm)
    else:
        wlm = np.imag(wlm)
    fig = plt.subplots(1, 1, figsize=figsize)
    # plt.plot(slm, 'k', label='Scaling fct')
    for j in range(J_min, J_max + 1):  # J_min <= j <= J_max
        if m is None:  # Axisym filters
            plt.plot(wlm[j, :], label=f'{j=}')
        else:  # Directionnal filters
            plt.plot(wlm[j, :, m], label=f'{j=}')
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'Filters $\Psi^j_{\ell 0}$')
    plt.xscale('log', base=2)
    plt.legend(loc='upper left')
    return fig


def plot_alm(flm, vmin=None, vmax=None, lmin=None, lmax=None, mmin=None, mmax=None,
             cmap='viridis', figsize=(12, 6), plot_only_real_part=False):
    """
    Plot the flm in the (l, m) plane.
    flm: array
        2D array [L, 2L-1] or [L, L]
    """
    L = flm.shape[0]

    if flm.shape[1] == L:
        flm = sphlib.make_flm_full(flm, L)  # [L, 2L-1]

    def for_all_plots(ax):
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(r'$m$')
        ax.set_xlim(lmin, lmax)
        ax.set_ylim(mmin, mmax)
        ax.plot(np.arange(L + 1), np.arange(L + 1), 'white')
        ax.plot(np.arange(L + 1), -np.arange(L + 1), 'white')
        #ax.grid()

    if plot_only_real_part:
        fig, (ax0) = plt.subplots(1, 1, figsize=figsize)
        im0 = ax0.imshow(np.real(flm).T, origin='lower', extent=(0, L, -L, L), cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=ax0)
        ax0.set_title('Real part')
        for_all_plots(ax0)
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

        im0 = ax0.imshow(np.real(flm).T, origin='lower', extent=(0, L, -L, L), cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=ax0)
        ax0.set_title('Real part')
        for_all_plots(ax0)

        im1 = ax1.imshow(np.imag(flm).T, origin='lower', extent=(0, L, -L, L), cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('Imaginary part')
        for_all_plots(ax1)

        fig.tight_layout()
    return fig


def plot_scatcov_coeffs(S1, P00, C01, C11, name=None, hold=True, color='blue', ls='-', marker=''):

    if name is None:
        name = ''

    if hold:
        plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.plot(np.real(S1), color=color, label=f'{name}', ls=ls, marker=marker)
    plt.title(r'$S_1$')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(np.real(P00), color=color, label=f'{name}', ls=ls, marker=marker)
    plt.title(r'$P_{00}$')

    plt.subplot(2, 2, 3)
    plt.plot(np.real(C01), color=color, label=f'{name} ' + r'$C_{01}$', ls=ls, marker=marker)
    plt.title(r'$C_{01}$')

    plt.subplot(2, 2, 4)
    plt.plot(np.real(C11), color=color, label=f'{name} ' + r'$C_{11}$', ls=ls, marker=marker)
    plt.title(r'$C_{11}$')
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