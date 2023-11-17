"""Classes for various cosmological species."""
from abc import ABC, abstractmethod
import numpy as np
import natpy as nat
from typing import Callable


class Species(ABC):
    def __init__(self,
                 label: str,
                 index: str):
        self.label = label
        self.index = index

    @abstractmethod
    def get_bg_energy_density(self):
        pass

    @abstractmethod
    def get_bg_pressure_density(self):
        pass


class PerfectFluid(Species):
    def __init__(self,
                 label: str,
                 index: str,
                 w: float,
                 Omega0: float,
                 H0: float):
        self.w = w
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label=label, index=index)

    def get_bg_energy_density(self, lna: float) -> float:
        return (self.H0**2)*(self.Omega0/np.exp(3.*(1. + self.w)*lna))

    def get_bg_pressure_density(self, lna: float) -> float:
        return self.w*self.get_bg_energy_density(lna)


class CDM(PerfectFluid):
    def __init__(self, Omega0: float, H0: float) -> None:
        super().__init__(label='Cold Dark Matter',
                         index='cdm',
                         w=0.,
                         Omega0=Omega0,
                         H0=H0)


class Photons(PerfectFluid):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Photons',
                         index='g',
                         w=1./3.,
                         Omega0=Omega0,
                         H0=H0)


class MasslessNeutrinos(PerfectFluid):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Massless neutrinos',
                         index='nu',
                         w=1./3.,
                         Omega0=Omega0,
                         H0=H0)


class Baryons(PerfectFluid):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Baryons',
                         index='b',
                         w=0.,
                         Omega0=Omega0,
                         H0=H0)


class DarkEnergy(PerfectFluid):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Dark energy',
                         index='Lambda',
                         w=-1.,
                         Omega0=Omega0,
                         H0=H0)


class ScalarFieldDarkMatter(Species):
    def __init__(self,
                 label: str,
                 index: str,
                 m_eV: float,
                 f: float,
                 theta_ini: float,
                 theta_der_ini: float):
        self.m_eV = m_eV
        self.m = nat.convert(m_eV*nat.eV, nat.Mpc**-1).value
        self.f = f
        self.theta_ini = theta_ini
        self.theta_der_ini = theta_der_ini
        super().__init__(label=label, index=index)

    def get_bg_energy_density(self, theta: float, theta_der: float, H: float
                              ) -> float:
        return (self.f**2)*(0.5*(H*theta_der)**2
                            + (self.m**2)*self.U(theta))/3.

    def get_bg_pressure_density(self, theta: float, theta_der: float, H: float
                                ) -> float:
        return (self.f**2)*(0.5*(H*theta_der)**2
                            - (self.m**2)*self.U(theta))/3.

    @abstractmethod
    def U(self, theta) -> float:
        pass

    @abstractmethod
    def Up(self, theta) -> float:
        pass

    @abstractmethod
    def Upp(self, theta) -> float:
        pass

    def get_precise_ics(self, z_ini: float) -> tuple[float, float]:
        # TODO: For now we just return the initial conditions.
        return self.theta_ini, self.theta_der_ini


class FreeScalarFieldDarkMatter(ScalarFieldDarkMatter):
    def __init__(self,
                 m_eV: float,
                 theta_ini: float,
                 theta_der_ini: float,
                 f: float = 1.):
        super().__init__(label='Free Scalar Field Dark Matter',
                         index='fscf',
                         m_eV=m_eV,
                         f=f,
                         theta_ini=theta_ini,
                         theta_der_ini=theta_der_ini)

    def U(self, theta: float) -> float:
        return 0.5*(theta**2)

    def Up(self, theta: float) -> float:
        return theta

    def Upp(self, theta: float) -> float:
        return 1.
