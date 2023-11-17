"""Background module."""
import numpy as np
import natpy as nat
from lya.species import Species
import lya.routines as routines
from scipy.integrate import solve_ivp, OdeSolution


def hubble(rho_arr: list[float]) -> float:
    return np.sqrt(np.sum(rho_arr))


def get_hubble(rho_tot: float) -> float:
    return np.sqrt(rho_tot)


def get_densities_pf(Omega0: float, lna: float, H0: float,
                     w: float) -> tuple[float, float]:
    # return (H0**2)*Omega0/np.exp(w*lna), w*(H0**2)*Omega0/np.exp(w*lna)
    return ((H0**2)*Omega0/np.exp(3.*(1. + w)*lna),
            w*(H0**2)*Omega0/np.exp(3.*(1. + w)*lna))


def initialize(H0: float, Omega_g0: float, Omega_nu0: float, Omega_cdm0: float,
               Omega_b0: float, Omega_Lambda0: float | None = None,
               Omega_scf0: float | None = None
               ) -> dict:
    # Either Omega_Lambda0 or Omega_scf0 should be given
    if (Omega_Lambda0 is None) and (Omega_scf0 is None):
        raise ValueError('Either `Omega_Lambda0` or `Omega_scf0` should be given!')

    # If Omega_Lambda0 is None, get it from the budget equation
    if Omega_Lambda0 is None:
        Omega_Lambda0 = (1. - Omega_g0 - Omega_nu0
                         - Omega_cdm0 - Omega_b0 - Omega_scf0)

    # If Omega_scf0 is None, get it from the budget equation
    if Omega_scf0 is None:
        Omega_scf0 = (1. - Omega_g0 - Omega_nu0
                      - Omega_cdm0 - Omega_b0 - Omega_Lambda0)

    # Check whether the budget equation is satisfied
    Omega_tot = Omega_g0 + Omega_nu0 + Omega_cdm0 + Omega_b0 + Omega_Lambda0
    assert np.isclose(Omega_tot, 1.)

    return {
        'has_photon': Omega_g0 != 0.,
        'has_massless_neutrino': Omega_nu0 != 0.,
        'has_cdm': Omega_cdm0 != 0.,
        'has_baryons': Omega_b0 != 0.,
        'has_scf': Omega_scf0 != 0.,
        'Omega_g0': Omega_g0,
        'Omega_nu0': Omega_nu0,
        'Omega_cdm0': Omega_cdm0,
        'Omega_b0': Omega_b0,
        'Omega_Lambda0': Omega_Lambda0,
        'H0': H0
    }


def get_total_energy_density_and_eos(bg: dict, lna: float
                                     ) -> tuple[float, float]:
    rho_tot = 0.
    p_tot = 0.

    # Photons
    rho_g, p_g = get_densities_pf(bg['Omega_g0'], lna, bg['H0'], 1./3.)
    rho_tot += rho_g
    p_tot += p_g

    # Massless neutrinos
    rho_nu, p_nu = get_densities_pf(bg['Omega_nu0'], lna, bg['H0'], 1./3.)
    rho_tot += rho_nu
    p_tot += p_nu

    # Cold dark matter
    rho_cdm, p_cdm = get_densities_pf(bg['Omega_cdm0'], lna, bg['H0'], 0.)
    rho_tot += rho_cdm
    p_tot += p_cdm

    # Baryons
    rho_b, p_b = get_densities_pf(bg['Omega_b0'], lna, bg['H0'], 0.)
    rho_tot += rho_b
    p_tot += p_b

    # Dark energy
    rho_L, p_L = get_densities_pf(bg['Omega_Lambda0'], lna, bg['H0'], -1.)
    rho_tot += rho_L
    p_tot += p_L
    return rho_tot, p_tot


def solve(bg: dict, z_ini: float = 1e14, z_fin: float = 0., **solve_ivp_args):
    # Get the initial ln_a
    lna_ini = np.log((1. + z_ini)**-1)

    # Get the initial energy density
    rho_ini = get_total_energy_density_and_eos(bg, lna_ini),

    # Get the initial Hubble
    H_ini = get_hubble(rho_ini)

    # Get the initial conditions for the conformal time tau, and
    # physical time t.
    tau_ini = np.exp(-lna_ini)/H_ini
    t_ini = 0.5/H_ini

    # Define the RHS of the system, i.e. the derivatives
    def rhs(lna, y):
        # y[0]: tau, y[1]: t
        rho_tot = get_total_energy_density_and_eos(bg, lna_ini),
        H = get_hubble(rho_tot)
        return [np.exp(-lna)/H, 1./H]

    # Integrate
    sol = solve_ivp(rhs, [lna_ini, np.log(1./(1. + z_fin))],
                    y0=[tau_ini, t_ini], **solve_ivp_args)

    return sol


def finalize(bg: dict, sol: OdeSolution):
    # Dictionary that will hold the calculated values for the energy densities
    bg['rho'] = {}
    bg['rho']['rho_g'] = np.zeros(len(sol.t))
    bg['rho']['rho_nu'] = np.zeros(len(sol.t))
    bg['rho']['rho_cdm'] = np.zeros(len(sol.t))
    bg['rho']['rho_b'] = np.zeros(len(sol.t))
    bg['rho']['rho_Lambda'] = np.zeros(len(sol.t))

    # Dictionary that will hold the calculated values for the pressure densities
    bg['pre'] = {}
    bg['pre']['pre_g'] = np.zeros(len(sol.t))
    bg['pre']['pre_nu'] = np.zeros(len(sol.t))
    bg['pre']['pre_cdm'] = np.zeros(len(sol.t))
    bg['pre']['pre_b'] = np.zeros(len(sol.t))
    bg['pre']['pre_Lambda'] = np.zeros(len(sol.t))

    # Calculated values of the Hubble
    bg['H'] = np.zeros(len(sol.t))

    # Fill the arrays
    for i, lna in enumerate(sol.t):
        bg['rho']['rho_g'][i], bg['pre']['pre_g'] = get_densities_pf(bg['Omega_g0'], lna, bg['H0'], 1./3.)
        bg['rho']['rho_nu'][i], bg['pre']['pre_nu'] = get_densities_pf(bg['Omega_nu0'], lna, bg['H0'], 1./3.)
        bg['rho']['rho_cdm'][i], bg['pre']['pre_cdm'] = get_densities_pf(bg['Omega_cdm0'], lna, bg['H0'], 0.)
        bg['rho']['rho_b'][i], bg['pre']['pre_b'] = get_densities_pf(bg['Omega_b0'], lna, bg['H0'], 0.)
        bg['rho']['rho_Lambda'][i], bg['pre']['pre_Lambda'] = get_densities_pf(bg['Omega_Lambda0'], lna, bg['H0'], -1.)

        rho_tot = (bg['rho']['rho_g'][i] + bg['rho']['rho_nu'][i]
                   + bg['rho']['rho_cdm'][i] + bg['rho']['rho_b'][i]
                   + bg['rho']['rho_Lambda'][i])
        bg['H'][i] = get_hubble(rho_tot)

    # Other calculated parameters
    bg['sol'] = sol
    bg['z'] = 1./np.exp(sol.t) - 1.
    bg['tau'] = sol.y[0]
    bg['t'] = sol.y[1]

    # We also need an interpolating function for H(tau)
    bg['interps'] = {}
    bg['interps']['H_vs_tau'] = routines.log_cubic_spline(bg['tau'], bg['H'])
    bg['interps']['a_vs_tau'] = routines.log_cubic_spline(bg['tau'], np.exp(sol.t))

    return bg


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
