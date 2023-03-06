from os.path import dirname
import numpy as np
import matplotlib.pyplot as plt
import pysm3
import astropy.units as u
import healpy as hp
from scipy.interpolate import interp2d

# import s2fft

# To open earth map
from PIL import Image
from matplotlib.image import pil_to_array

# __all__ = ['Spherical_Compute']

# """
# This library is made to
# - produce Healpix sky map
# - define mathematical operations on the sphere (mean, scalar product...)
# - plot the flm
# """

############ MAKING SKY MAPS


def make_hpx_planet(nside, planet, dirmap=dirname(dirname(__file__)) + '/texture_maps',
                    interp=True, normalize=False, nest=False):
    """
    Create a Healpix map from a JPG.
    For now, there are 4 options: CMB, Dust, random noise or the Earth map.
    Parameters
    ----------
    nside: int
        Nside parameter from Healpix. The number of pixel is 12xNside^2.
        It must be a power of 2.
    planet: str
        Name of the planet. Keywords allowed are: 'earth', 'sun', 'moon', 'mercury', 'venus', 'jupiter', 'ceres'
    dirmap: str
        Directory where planet maps are stored.
    interp: bool
         If True, make an interpolation of the 2D array.
    normalize: bool
        If True, the mean of the map is set to 0 and the STD to 1.
    nest: bool
        If True return a Healpix map in NEST ordering instead of RING ordering. False by default.
    Returns
    -------
    map: the Healpix map.
    """
    grayscale_pil_image = Image.open(dirmap + f'/{planet}.jpg').convert("L")
    image_array = pil_to_array(grayscale_pil_image)
    theta = np.linspace(0, np.pi, num=image_array.shape[0])[:, None]
    phi = np.linspace(0, 2 * np.pi, num=image_array.shape[1])
    npix = hp.nside2npix(nside)

    if interp:
        f = interp2d(theta, phi, image_array.T, kind='cubic')
        map = np.zeros(npix)
        for p in range(npix):
            th, ph = hp.pix2ang(nside=nside, ipix=p)
            map[p] = f(th, ph)
    else:
        pix = hp.ang2pix(nside, theta, phi)
        map = np.zeros(npix, dtype=np.double)
        map[pix] = image_array
    # Convert float64 to float32
    map = np.array(map, dtype=np.float32)
    if normalize:  # Normalize: mean=0 and std=1
        map -= np.mean(map)
        map /= np.std(map)
    print(f'Mean and STD: {np.mean(map):.3f} and {np.std(map):.3f}')

    # Convert from RING to NEST ordering
    if nest:
        map = hp.reorder(map, r2n=True)

    return map


def make_hpx_sky(nside, sky_type, normalize=False, nest=False):
    """
    Create a Healpix map.
    For now, there are 3 options: CMB, Dust or random noise.
    Parameters
    ----------
    nside: int
        Nside parameter from Healpix. The number of pixel is 12xNside^2.
        It must be a power of 2.
    sky_type: str
        Type of sky. Keywords allowed are: 'cmb', 'dust', 'noise'
    normalize: bool
        If True, the mean of the map is set to 0 and the STD to 1.
    nest: bool
        If True return a Healpix map in NEST ordering instead of RING ordering. False by default.
    Returns
    -------
    map: the Healpix map.
    """

    if sky_type == 'noise':  # White noise
        map = np.random.random(size=12 * nside ** 2)

    elif sky_type == 'cmb':  # CMB sky
        sky = pysm3.Sky(nside=nside, preset_strings=["c1"], output_unit="K_CMB")
        cmb_maps = sky.get_emission(freq=np.array(150) * u.GHz)
        map = cmb_maps[0, :].value  # Take only intensity and remove unit

    elif sky_type == 'dust':  # Dust sky
        sky = pysm3.Sky(nside=nside, preset_strings=["d1"], output_unit="K_CMB")
        dust_maps = sky.get_emission(freq=np.array(400) * u.GHz)
        map = dust_maps[0, :].value  # Take only intensity and remove unit

    else:
        raise ValueError('sky_type argument has a wrong value.')
    # Convert float64 to float32
    map = np.array(map, dtype=np.float32)
    if normalize:  # Normalize: mean=0 and std=1
        map -= np.mean(map)
        map /= np.std(map)
    print(f'Mean and STD: {np.mean(map):.3f} and {np.std(map):.3f}')

    # Convert from RING to NEST ordering
    if nest:
        map = hp.reorder(map, r2n=True)

    return map



# ############ AVERAGE, VARIANCE AND PS ON SPHERE
# class Spherical_Compute():
#     """
#     This class defines the averages and the scalar products (SP) on map space or alm space.
#     """

#     def __init__(self, nside):
#         """
#         """
#         self.nside = nside
#         self.npix = 12 * nside ** 2
#         self.lmax = 3 * self.nside - 1  # We use the Healpix default value
#         self.L = self.lmax + 1
#         self.l_hpx, self.m_hpx = hp.Alm.getlm(self.lmax)

#     def forward_jax(self, map_hpx):
#         return s2fft.forward_jax(map_hpx, L=self.L, nside=self.nside, sampling='healpix')

#     def inverse_jax(self, alm_hpx):
#         return s2fft.inverse_jax(alm_hpx, L=self.L, nside=self.nside, sampling='healpix')

#     def average_on_sphere_hpx(self, map_hpx):
#         """
#         Average of the pixel values for a Healpix map:
#             mean = sum_p(map_p) / Npix where p is the pixel index.
#         This is equivalent to np.mean.
#         Parameters
#         ----------
#         map_hpx: numpy array
#             Healpix map [12*Nside**2].
#         Returns
#         -------
#         mean
#         """
#         mean = np.sum(map_hpx) * (4 * np.pi / self.npix)
#         mean /= (4 * np.pi)
#         return mean

#     def average_with_a00(self, alm):
#         """
#         Compute the average of a map using the (l=0, m=0) term.
#         This function can be used with HPX and MW alms.
#         Parameters
#         ----------
#         alm: tensor
#             alm coefficients (HPX or MW).
#             The tensor array can have multiple dimensions, for example [Nimg, Nalm] but alm must be in the last column.

#         Returns
#         -------
#         The average of the map.
#         """
#         return alm[..., 0] / (2 * np.sqrt(np.pi))

#     def average_alm_hpx(self, alm_hpx):
#         """
#         Average alm ordered as in Healpix.
#         The factor 4pi is there to match the definition of the average in pixel space.
#         As Healpix does not store coefficients for m<0, we sum the alm twice and we remove the m=0 terms once.
#         Parameters
#         ----------
#         alm_hpx: array
#             alm coefficient ordered as in Healpix.
#         Returns
#         -------
#         avg: the average of the alm.
#         """
#         avg = (np.sum(alm_hpx) * 2 - np.sum(alm_hpx[:self.lmax])) / (4 * np.pi)
#         return avg

#     def variance_on_sphere_hpx(self, map_hpx):
#         # Subtract the mean in map space
#         map_hpx_mean0 = map_hpx - self.average_on_sphere_hpx(map_hpx)
#         # Compute the variance
#         var = self.average_on_sphere_hpx(map_hpx_mean0 * np.conj(map_hpx_mean0))
#         return var

#     def variance_alm_hpx(self, alm_hpx):
#         """
#         As Healpix does not store coefficients for m<0, we sum the alm twice and we remove the m=0 terms once.
#         Parameters
#         """
#         # Do not sum the (l, m) = (0, 0) term: equivalent to subtract the mean
#         var = (np.sum(alm_hpx[1:] * np.conj(alm_hpx[1:])) * 2
#                - np.sum(alm_hpx[1:self.lmax] * np.conj(alm_hpx[1:self.lmax]))
#                ) / (4 * np.pi)
#         return var

#     def ps_on_sphere_hpx(self, map_hpx0, map_hpx1):
#         ps = self.average_on_sphere_hpx(map_hpx0 * np.conj(map_hpx1))
#         return ps

#     def ps_alm_hpx(self, alm_hpx0, alm_hpx1):
#         ps = self.average_alm_hpx(alm_hpx0 * np.conj(alm_hpx1))
#         return ps

#     def plot_alm(self, alm, s=10, vmin=None, vmax=None, lmin=None, lmax=None, mmin=None, mmax=None,
#                  cmap='viridis', figsize=(12, 6), fontsize=16, real_part_only=False):
#         """
#         Plot the alm in the (l, m) plane. By default, it plots real part, imaginary part and amplitude.
#         Parameters
#         ----------
#         alm: tensor
#             alm hpx coefficients.
#         s: int
#             Size of the points.
#         real_part_only: bool
#             if True, only plot the real part.

#         Returns
#         -------

#         """
#         ell, em = self.l_hpx, self.m_hpx

#         if real_part_only:
#             fig, ax = plt.subplots(1, 1, figsize=figsize)
#             im0 = ax.scatter(ell, em, c=np.real(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
#             fig.colorbar(im0, ax=ax)
#             ax.set_xlabel(r'$\ell$', fontsize=fontsize)
#             ax.set_ylabel('m', fontsize=fontsize)
#             ax.axis('scaled')
#             ax.set_xlim(lmin, lmax)
#             ax.set_ylim(mmin, mmax)
#             ax.set_title('Real part', fontsize=fontsize)
#             fig.tight_layout()
#         else:
#             fig, axs = plt.subplots(1, 3, figsize=figsize)
#             axs = np.ravel(axs)
#             im0 = axs[0].scatter(ell, em, c=np.real(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
#             fig.colorbar(im0, ax=axs[0])
#             axs[0].set_xlabel(r'$\ell$', fontsize=fontsize)
#             axs[0].set_ylabel('m', fontsize=fontsize)
#             axs[0].set_xlim(lmin, lmax)
#             axs[0].set_ylim(mmin, mmax)
#             axs[0].set_title('Real part', fontsize=fontsize)

#             im1 = axs[1].scatter(ell, em, c=np.imag(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
#             fig.colorbar(im1, ax=axs[1])
#             axs[1].set_xlabel(r'$\ell$', fontsize=fontsize)
#             axs[1].set_ylabel('m', fontsize=fontsize)
#             axs[1].set_xlim(lmin, lmax)
#             axs[1].set_ylim(mmin, mmax)
#             axs[1].set_title('Im part', fontsize=fontsize)

#             im2 = axs[2].scatter(ell, em, c=np.abs(alm), s=s, cmap=cmap, vmin=vmin, vmax=vmax)
#             fig.colorbar(im2, ax=axs[2])
#             axs[2].set_xlabel(r'$\ell$', fontsize=fontsize)
#             axs[2].set_ylabel('m', fontsize=fontsize)
#             axs[2].set_xlim(lmin, lmax)
#             axs[2].set_ylim(mmin, mmax)
#             axs[2].set_title('Amplitude', fontsize=fontsize)
#             fig.tight_layout()
#         return fig
