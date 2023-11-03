"""Modules for defining the cosmological model."""
import natpy as nat
import numpy as np
from alpfrag.constants import M_PLANCK_REDUCED
import lya.background as bg

# This sets hbar, c, and kB (Boltzmanns's constant) to one
nat.set_active_units('HEP')


class Cosmology:
    def __init__(self, *,
                 T_cmb: float = 2.7255,
                 omega_b0: float = 0.02238280,
                 N_ur: float = 3.044,
                 omega_cdm0: float = 0.1201075,
                 h: float = 0.67810):
        self.T_cmb = T_cmb
        self.omega_b0 = omega_b0
        self.N_ur = N_ur
        self.omega_cdm0 = omega_cdm0
        self.h = h

    @property
    def rho_crit0_class(self):
        return (self.H0(nat.Mpc**-1)**2).value

    @property
    def Omega_g0(self):
        T_cmb_GeV = nat.convert(self.T_cmb*nat.K, nat.GeV).value
        H0_GeV = self.H0(nat.GeV).value
        mpl_GeV = M_PLANCK_REDUCED.value
        return (np.pi**2)*(T_cmb_GeV**4)/(45.*((mpl_GeV**2)*(H0_GeV**2)))

    @property
    def omega_g0(self):
        return self.Omega_g*(self.h**2)

    @property
    def Omega_nu0(self):
        return (7./8.)*self.N_ur*((4./11.)**(4./3.))*self.Omega_g0

    @property
    def Omega_r0(self):
        return self.Omega_g0 + self.Omega_nu0

    @property
    def omega_nu0(self):
        return self.Omega0_nu*(self.h**2)

    @property
    def Omega_cdm0(self):
        return self.omega_cdm0/(self.h**2)

    @property
    def Omega_b0(self):
        return self.omega_b0/(self.h**2)

    @property
    def Omega_Lambda0(self):
        # Determine from the budget equation
        return (1. - self.Omega_g0 - self.Omega_nu0
                - self.Omega_cdm0 - self.Omega_b0)

    def H0(self, target_unit):
        return nat.convert(100.*self.h*nat.km/(nat.s*nat.Mpc), target_unit)

    def compute_background(self, z_ini: float = 1e14, z_fin: float = 0.,
                           method: str = 'DOP853',
                           rtol: float = 1e-12, atol: float = 1e-12):
        bg_dict = bg.initialize(self.Omega_g0, self.Omega_nu0, self.Omega_cdm0,
                                self.Omega_b0, self.Omega_Lambda0,
                                self.H0(nat.Mpc**-1).value)
        bg_sol = bg.solve(bg_dict, z_ini, z_fin, method=method, rtol=rtol,
                          atol=atol)
        self.background = bg.finalize(bg_dict, bg_sol)


if __name__ == "__main__":
    cosmo = Cosmology()
    print("{:e}".format(cosmo.omega_g()))
