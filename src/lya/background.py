"""Background module."""
import numpy as np
from lya.species import Species
from scipy.integrate import solve_ivp


def hubble(rho_arr: list[float]) -> float:
    return np.sqrt(np.sum(rho_arr))


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

    if sol.success:
        return True
    else:
        raise RuntimeError(f"Integration failed. {sol.message}")
