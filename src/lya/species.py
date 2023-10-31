"""Classes for various cosmological species."""
from abc import ABC, abstractmethod
import numpy as np


class Species(ABC):
    def __init__(self,
                 label: str,
                 index: str):
        self.label = label
        self.index = index

    @abstractmethod
    def get_bg_energy_density(self, lna: float, *args):
        pass


class CDM(Species):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Cold Dark Matter', index='cdm')

    def get_bg_energy_density(self, lna: float):
        return (self.H0**2)*self.Omega0/np.exp(3.*lna)


class Photons(Species):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Photons', index='g')

    def get_bg_energy_density(self, lna: float):
        return (self.H0**2)*self.Omega0/np.exp(4.*lna)


class MasslessNeutrinos(Species):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Massless neutrinos', index='nu')

    def get_bg_energy_density(self, lna: float):
        return (self.H0**2)*self.Omega0/np.exp(4.*lna)


class Baryons(Species):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Baryons', index='b')

    def get_bg_energy_density(self, lna: float):
        return (self.H0**2)*self.Omega0/np.exp(3.*lna)


class DarkEnergy(Species):
    def __init__(self, Omega0: float, H0: float) -> None:
        self.Omega0 = Omega0
        self.H0 = H0
        super().__init__(label='Dark energy', index='de')

    def get_bg_energy_density(self, lna: float):
        return (self.H0**2)*self.Omega0/np.exp(0.*lna)
