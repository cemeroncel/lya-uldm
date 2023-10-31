"""Background module."""
import numpy as np
import natpy as nat
from lya.species import Species
from scipy.integrate import solve_ivp


def hubble(rho_arr: list[float]) -> float:
    return np.sqrt(np.sum(rho_arr))


def initialize(Omega_g0: float, Omega_nu0: float, Omega_cdm0: float,
               Omega_b0: float, Omega_Lambda0: float) -> dict:
    Omega_tot = Omega_g0 + Omega_nu0 + Omega_cdm0 + Omega_b0 + Omega_Lambda0
    assert np.isclose(Omega_tot, 1.)
    return {}


def integrate(species_list: list[Species], z_ini: float = 1e14,
              z_fin: float = 0., **solve_ivp_args):
    # Number of species in the universe
    number_of_species = len(species_list)

    # Get the initial ln_a
    lna_ini = np.log((1. + z_ini)**-1)

    # Helper function to update the array of the energy densities
    def update_rho_arr(lna):
        rho_arr = np.zeros(number_of_species)
        for i, species in enumerate(species_list):
            rho_arr[i] = species.get_bg_energy_density(lna)
        return rho_arr

    # Create the array which will hold the energy densities and
    # initialize it with the initial conditions
    rho_arr = update_rho_arr(lna_ini)

    # Get the initial conditions for the conformal time tau, and
    # physical time t.
    tau_ini = np.exp(-lna_ini)/hubble(rho_arr)
    t_ini = 0.5/hubble(rho_arr)

    # Define RHS of the system, i.e. the derivaties
    def rhs(lna, y):
        # y[0]: tau, y[1]: t
        rho_arr = update_rho_arr(lna)
        H = hubble(rho_arr)
        return [np.exp(-lna)/H, 1./H]

    # Integrate
    sol = solve_ivp(rhs, [lna_ini, np.log(1./(1. + z_fin))],
                    y0=[tau_ini, t_ini], **solve_ivp_args)

    # Create the dictionary of the energy densities and populate them
    rho = dict()
    for species in species_list:
        rho[species.index] = species.get_bg_energy_density(sol.t)

    # Create the Hubble array
    H = np.zeros(len(sol.t))
    for i, lna in enumerate(sol.t):
        rho_arr = update_rho_arr(lna)
        H[i] = hubble(rho_arr)

    if sol.success:
        return {
            'sol': sol,
            'z+1': 1./np.exp(sol.t),
            'tau': sol.y[0],
            't': sol.y[1],
            't_Gyr': nat.convert(sol.y[1]*nat.Mpc, nat.Gyr).value,
            'rho': rho,
            'H': H
        }
    else:
        raise RuntimeError(f"Integration failed. {sol.message}")
