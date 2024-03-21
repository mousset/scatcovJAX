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
from astropy.io import fits
#from astropy.utils.data import get_pkg_data_filename


def normalize_map(I):
    """ Normalize the map I: mean=0 and std=1."""
    I -= np.nanmean(I)
    I /= np.nanstd(I)
    print(f'Mean and STD: {np.mean(I):.3f} and {np.std(I):.3f}')
    return I


def sort_CosmoGrid_maps(data_LSS):
    sort = np.argsort(data_LSS['shell_info'])

    shell_info = data_LSS['shell_info'][sort]
    shells = data_LSS['shells'][sort]

    return shell_info, shells


def get_mean_redshift(shell_info):
    nmaps = shell_info.size
    z_mean = np.zeros(nmaps)
    for i in range(nmaps):
        lower_z = shell_info[i][2]
        upper_z = shell_info[i][3]
        z_mean[i] = (lower_z + upper_z) / 2

    return z_mean


def make_CosmoGrid_sky(L, dirmap, run=0, idx_z=10, sampling='mw', nest=False, normalize=False, reality=True):
    """

    Parameters
    ----------
    L
    dirmap
    run: int
        Index of the run.
    idx_z: int
        Index of the map from 0 to 68 corresponding to a given redshift z.
    sampling
    nest
    normalize
    reality

    Returns
    -------

    """
    nside = int(L / 2)

    ### Get the Healpix map
    # Get all the maps sort by redshift
    data_LSS = np.load(dirmap + f'baryonified_shells_run{run:04}.npz')
    shell_info, shells = sort_CosmoGrid_maps(data_LSS)
    # Take the map
    I = shells[idx_z, :]
    z = get_mean_redshift(shell_info)[idx_z]
    print(f'Map at redshift {z=}')
    # From nside=512 to nside=L/2
    I = hp.ud_grade(I, nside_out=nside)
    # Take the log
    I = np.log(I + 0.001)

    if normalize:  # Normalize: mean=0 and std=1
        I = normalize_map(I)

    # Convert from RING to NEST ordering in case of healpix sampling
    if nest:
        I = hp.reorder(I, r2n=True)

    # SHT inverse transform at L
    Ilm = s2fft.forward_jax(I, L, sampling='healpix', nside=nside, reality=reality)  # [L, 2L-1]

    if sampling == 'mw':
        ### Make a MW map
        I = s2fft.inverse_jax(Ilm, L, sampling='mw', nside=None, reality=reality)  # [Ntheta, 2Ntheta-1]
        print(f'Mean and STD: {np.mean(I):.3f} and {np.std(I):.3f}')
    elif sampling == 'healpix':
        ### Inverse transform to kill small scales and have same power in I and Ilm
        I = s2fft.inverse_jax(Ilm, L=L, nside=nside, reality=reality, sampling='healpix')
    # Get only positive m
    if reality:
        Ilm = Ilm[:, L - 1:]  # [L, L]

    return I, Ilm


def make_NASAsimu_sky(L, mapfile, sampling='mw', nest=False, normalize=False, reality=True, sky='lensing'):
    """

    Parameters
    ----------
    L
    dirmap
    sampling
    nest
    normalize
    reality

    Returns
    -------

    """
    nside = int(L / 2)

    ### Get the Healpix map
    image_file = fits.open(mapfile)
    I = hp.read_map(image_file, 0, h=False)

    # From nside=4096 to nside=L/2
    I = hp.ud_grade(I, nside_out=nside)
    # Take the log
    if sky == 'lensing':
        I = np.log(I + 0.0001)  # For Lensing
    elif sky == 'tsz':
        I = np.log(I)  # For tSZ

    if normalize:  # Normalize: mean=0 and std=1
        I = normalize_map(I)

    # Convert from RING to NEST ordering in case of healpix sampling
    if nest:
        I = hp.reorder(I, r2n=True)

    # SHT inverse transform at L
    Ilm = s2fft.forward_jax(I, L, sampling='healpix', nside=nside, reality=reality)  # [L, 2L-1]

    if sampling == 'mw':
        ### Make a MW map
        I = s2fft.inverse_jax(Ilm, L, sampling='mw', nside=None, reality=reality)  # [Ntheta, 2Ntheta-1]
        print(f'Mean and STD: {np.mean(I):.3f} and {np.std(I):.3f}')
    elif sampling == 'healpix':
        ### Inverse transform to kill small scales and have same power in I and Ilm
        I = s2fft.inverse_jax(Ilm, L=L, nside=nside, reality=reality, sampling='healpix')
    # Get only positive m
    if reality:
        Ilm = Ilm[:, L - 1:]  # [L, L]

    return I, Ilm


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
        I = normalize_map(I)

    # Convert from RING to NEST ordering in case of healpix sampling
    if nest:
        I = hp.reorder(I, r2n=True)

    # SHT inverse transform at L
    Ilm = s2fft.forward_jax(I, L, sampling='healpix', nside=nside, reality=reality)  # [L, 2L-1]

    if sampling == 'mw':
        ### Make a MW map
        I = s2fft.inverse_jax(Ilm, L, sampling='mw', nside=None, reality=reality)  # [Ntheta, 2Ntheta-1]
        print(f'Mean and STD: {np.mean(I):.3f} and {np.std(I):.3f}')
    elif sampling == 'healpix':
        ### Inverse transform to kill small scales and have same power in I and Ilm
        I = s2fft.inverse_jax(Ilm, L=L, nside=nside, reality=reality, sampling='healpix')
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
        I = normalize_map(I)
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


def compute_ps(flm, reality=False):
    """Compute the angular power spectrum Cls = 1/(2l+1) Sum_m[|f_lm|^2]."""
    L = flm.shape[0]
    ell = np.arange(L)
    Cls = jnp.nansum(jnp.abs(flm) ** 2, axis=-1) / (2 * ell + 1)
    if reality:
        Cls = 2. * Cls - Cls[0]
    return Cls


def gaussian(x, mu, sigma):
    """ Return the normalized Gaussian with standard deviation sigma and mean mu. """
    return jnp.exp(-0.5 * ((x-mu) / sigma)**2)


def make_linear_filters(Nfilters, L):
    """
    We linearly span the l axis from 0 to L-1 with Gaussian with standard deviation equal to L//Nfilters
    and normalized with maximum equal to 1.
    """
    filter_lin = np.zeros((Nfilters, L))
    ll = np.arange(0, L)
    for j in range(Nfilters):
        filter_lin[j, :] = gaussian(ll, L//(Nfilters) * (j+1), L//Nfilters)
    return filter_lin


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

