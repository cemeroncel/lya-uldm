"""Modules for defining the cosmological model."""
import natpy as nat
import numpy as np
from alpfrag.constants import M_PLANCK_REDUCED

# This sets hbar, c, and kB (Boltzmanns's constant) to one
nat.set_active_units('HEP')


class Cosmology:
    def __init__(self, *,
                 T_cmb: float = 2.7255,
                 omega_b: float = 0.02238280,
                 N_ur: float = 3.044,
                 omega_dm: float = 0.1201075,
                 h: float = 0.67810):
        self.T_cmb = T_cmb
        self.omega_b = omega_b
        self.N_ur = N_ur
        self.omega_dm = omega_dm
        self.h = h

    def H0(self, target_unit):
        return nat.convert(100.*self.h*nat.km/(nat.s*nat.Mpc), target_unit)

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
    def omega_nu0(self):
        return self.Omega0_nu*(self.h**2)

    @property
    def Omega_dm0(self):
        return self.omega_dm/(self.h**2)

    @property
    def Omega_b0(self):
        return self.omega_b/(self.h**2)


if __name__ == "__main__":
    cosmo = Cosmology()
    print("{:e}".format(cosmo.omega_g()))
