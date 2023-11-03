"""Tests for the cosmology module."""


import numpy as np
import lya.cosmology as cosmology
from classy import Class


class TestCosmologyProperties:
    cosmo = cosmology.Cosmology()
    class_cosmo = Class()
    class_cosmo.set(
        {
            'T_cmb': cosmo.T_cmb,
            'omega_b': cosmo.omega_b0,
            'omega_cdm': cosmo.omega_cdm0,
            'h': cosmo.h,
        }
    )
    class_cosmo.compute()

    def test_Omega_g0(self):
        expected = self.class_cosmo.Omega_g()
        obtained = self.cosmo.Omega_g0
        assert np.isclose(expected, obtained)

    def test_Omega_r0(self):
        expected = self.class_cosmo.Omega_r()
        obtained = self.cosmo.Omega_r0
        assert np.isclose(expected, obtained)
