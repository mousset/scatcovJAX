import numpy as np
import matplotlib.pyplot as plt

from scatcovjax.Sphere_lib import Spherical_Compute
import scatcovjax.Wavelets_lib as wlib

__all__ = ['Scattering_sph']

"""
This library is made to
- compute the scattering covariance coefficients (axisym or directional)
- plot the coefficients

!!! TO DO:
- subsampling or not ?
- normalisation of the coeff by P00
- functions to plot the coeffs
- do as JM, return a class with the 4 coeffs ? 
"""


class Scattering_sph(Spherical_Compute):

    def __init__(self, nside, J, N, filter_set_alm):
        """

        Parameters
        ----------
        nside: int
        J: int
            Number of j scales (scale along l).
        N: int
            Number of orientation scales.
            N=1 for Axisym and must be odd for directional.
        filter_set_alm: tensor array
            alm of the wavelet. [J, Nalm]
        """
        super().__init__(nside)
        self.J, self.N = J, N

    def scat_cov_axi(self, data_alm):

        return

    def scat_cov_dir(self, data_alm):

        return
