from os.path import dirname
import numpy as np
import pysm3
import astropy.units as u
import healpy as hp
from scipy.interpolate import interp2d
import jax.numpy as jnp
import s2fft

# To open earth map
from PIL import Image
from matplotlib.image import pil_to_array


# """
# This library is made to
# - Load Healpix or MW sky map
# - define mathematical operations on the sphere (mean, scalar product...)
# """

############ MAKING SKY MAPS


# def make_hpx_planet(nside, planet, dirmap=dirname(dirname(__file__)) + '/texture_maps',
#                     interp=True, normalize=False, nest=False):
#     """
#     Create a Healpix map from a JPG.
#     For now, there are 4 options: CMB, Dust, random noise or the Earth map.
#     Parameters
#     ----------
#     nside: int
#         Nside parameter from Healpix. The number of pixel is 12xNside^2.
#         It must be a power of 2.
#     planet: str
#         Name of the planet. Keywords allowed are: 'earth', 'sun', 'moon', 'mercury', 'venus', 'jupiter', 'ceres'
#     dirmap: str
#         Directory where planet maps are stored.
#     interp: bool
#          If True, make an interpolation of the 2D array.
#     normalize: bool
#         If True, the mean of the map is set to 0 and the STD to 1.
#     nest: bool
#         If True return a Healpix map in NEST ordering instead of RING ordering. False by default.
#     Returns
#     -------
#     map: the Healpix map.
#     """
#     grayscale_pil_image = Image.open(dirmap + f'/{planet}.jpg').convert("L")
#     image_array = pil_to_array(grayscale_pil_image)
#     theta = np.linspace(0, np.pi, num=image_array.shape[0])[:, None]
#     phi = np.linspace(0, 2 * np.pi, num=image_array.shape[1])
#     npix = hp.nside2npix(nside)
#
#     if interp:
#         f = interp2d(theta, phi, image_array.T, kind='cubic')
#         map = np.zeros(npix)
#         for p in range(npix):
#             th, ph = hp.pix2ang(nside=nside, ipix=p)
#             map[p] = f(th, ph)
#     else:
#         pix = hp.ang2pix(nside, theta, phi)
#         map = np.zeros(npix, dtype=np.double)
#         map[pix] = image_array
#     # Convert float64 to float32
#     map = np.array(map, dtype=np.float32)
#     if normalize:  # Normalize: mean=0 and std=1
#         map -= np.mean(map)
#         map /= np.std(map)
#     print(f'Mean and STD: {np.mean(map):.3f} and {np.std(map):.3f}')
#
#     # Convert from RING to NEST ordering
#     if nest:
#         map = hp.reorder(map, r2n=True)
#
#     return map


def make_pysm_sky(L, sky_type, sampling='mw', nest=False, normalize=False, reality=True):
    """
    Create a Healpix or MW map.
    For now, there are 2 options: CMB or Dust.
    Parameters
    ----------
    sky_type: str
        Type of sky. Keywords allowed are: 'cmb', 'dust'
    normalize: bool
        If True, the mean of the map is set to 0 and the STD to 1.
    nest: bool
        If True return a Healpix map in NEST ordering instead of RING ordering. False by default.
    Returns
    -------
    map: the Healpix map.
    """
    # PySM gives healpix maps
    nside = int(L/2)
    if sky_type == 'cmb':  # CMB sky
        sky = pysm3.Sky(nside=nside, preset_strings=["c1"], output_unit="K_CMB")
        cmb_maps = sky.get_emission(freq=np.array(150) * u.GHz)
        I = cmb_maps[0, :].value  # Take only intensity and remove unit

    elif sky_type == 'dust':  # Dust sky
        sky = pysm3.Sky(nside=nside, preset_strings=["d1"], output_unit="K_CMB")
        dust_maps = sky.get_emission(freq=np.array(400) * u.GHz)
        I = dust_maps[0, :].value  # Take only intensity and remove unit
    else:
        raise ValueError('sky_type argument must be cmb or dust.')

    if normalize:  # Normalize: mean=0 and std=1
        I -= np.mean(I)
        I /= np.std(I)
    print(f'Mean and STD: {np.mean(I):.3f} and {np.std(I):.3f}')

    # Convert from RING to NEST ordering in case of healpix sampling
    if nest:
        I = hp.reorder(I, r2n=True)

    # SHT inverse transform at L
    Ilm = s2fft.forward_jax(I, L, sampling='healpix', nside=nside, reality=reality)  # [L, 2L-1]

    # Make a MW map
    if sampling == 'mw':
        I = s2fft.inverse_jax(Ilm, L, sampling='mw', nside=None, reality=reality)  # [Ntheta, 2Ntheta-1]
        print(f'Mean and STD: {np.mean(I):.3f} and {np.std(I):.3f}')
    # Get only positive m
    if reality:
        Ilm = Ilm[:, L - 1:]  # [L, L]

    return I, Ilm


def make_planet(L, planet, sampling='mw', nside=None, nest=False, dirmap=dirname(dirname(__file__)) + '/texture_maps',
                normalize=False, reality=True):
    """
    Make a planet map.
    If sampling='mw', the map is a 2D array [Ntheta, Nphi]=[L, 2L-1]
    """
    # Load the JPG map
    grayscale_pil_image = Image.open(dirmap + f'/{planet}.jpg').convert("L")
    I = pil_to_array(grayscale_pil_image).astype(np.float64)
    I = np.ascontiguousarray(I[:, :-1])  # Remove the last phi dimension to have [Ntheta, 2Ntheta -1] shape

    # SHT forward transform at L_temp=Ntheta
    L_temp, _ = I.shape
    Ilm = s2fft.forward_jax(I, L_temp, reality=reality)  # [L_temp, 2L_temp - 1]

    # Cut the Ilm at the L resolution
    Ilm = Ilm[:L, L_temp - L:L_temp + L - 1]  # [L, 2L-1]
    print(Ilm.shape)

    # SHT inverse transform at L
    I = s2fft.inverse_jax(Ilm, L, sampling=sampling, nside=nside, reality=reality)  # [Ntheta, 2Ntheta-1] or [Npix]
    if normalize:
        I -= np.mean(I)
        I /= np.std(I)
    # Convert from RING to NEST ordering in case of healpix sampling
    if nest:
        I = hp.reorder(I, r2n=True)

    # SHT forward transform at L
    Ilm = s2fft.forward_jax(I, L, sampling=sampling, nside=nside, reality=reality)  # [L, 2L-1]

    # Get only positive m
    if reality:
        Ilm = Ilm[:, L-1:]  # [L, L]

    return I, Ilm


def make_MW_lensing(L, dirmap=dirname(dirname(__file__)) + '/texture_maps/raw_data/',
                   normalize=False, reality=True):

    if L in [256, 350, 400, 512]:
        I = np.load(dirmap + f'CosmoML_shell_40_L_{L}.npy')
    elif L < 256:
        I = np.load(dirmap + f'CosmoML_shell_40_L_256.npy')
        Ilm = s2fft.forward_jax(I, 256, reality=reality)
        # Cut the Ilm at the L resolution
        Ilm = Ilm[:L, 256 - L:256 + L - 1]  # [L, 2L-1]
        # SHT inverse transform at L
        I = s2fft.inverse_jax(Ilm, L, reality=reality)  # [Ntheta, 2Ntheta-1]
    else:
        raise ValueError('Wrong L value.')

    if normalize:
        I -= np.nanmean(I)
        I /= np.nanstd(I)

    # SHT forward transform at L
    Ilm = s2fft.forward_jax(I, L, reality=reality)

    # Get only positive m
    if reality:
        Ilm = Ilm[:, L-1:]  # [L, L]

    return I, Ilm

############ OPERATIONS ON flm


def make_flm_full(flm_half, L):
    # Create and store signs
    msigns = (-1) ** jnp.arange(1, L)

    # Reflect and apply hermitian symmetry
    flm_full = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    flm_full = flm_full.at[:, L - 1:].set(flm_half)
    flm_full = flm_full.at[:, : L - 1].set(jnp.flip(jnp.conj(flm_full[:, L:]) * msigns, axis=-1))

    return flm_full


def compute_ps(flm):
    return jnp.sum(jnp.abs(flm)**2, axis=-1)

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

