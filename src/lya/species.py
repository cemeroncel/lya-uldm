"""Classes for various cosmological species."""
from abc import ABC, abstractmethod


class Species(ABC):
    def __init__(self,
                 label: str,
                 index: str):
        self.label = label
        self.index = index

    @abstractmethod
    def get_bg_energy_density(self, lna: float, *args):
        pass
