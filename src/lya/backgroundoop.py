"""Background module in OOP style."""
import numpy as np
import lya.species as lyas
from scipy.integrate import solve_ivp, OdeSolution


def get_hubble(rho_tot: float) -> float:
    return np.sqrt(rho_tot)


def get_total_densities(bg: dict, lna: float, y: list[float]):
    # First we calculate the densities of perfect fluids
    rho_pf = 0.
    pre_pf = 0.

    for s in bg['pf_species']:
        rho_pf += s.get_bg_energy_density(lna)
        pre_pf += s.get_bg_pressure_density(lna)

    # If scalar field is present, we add the correction
    if bg['has_scf']:
        s = bg['scf_species'][0]
        theta = y[bg['ind']['scf_theta']]
        theta_der = y[bg['ind']['scf_theta_der']]
        rho_tot = ((rho_pf + ((s.m*s.f)**2)*s.U(theta)/3.)
                   / (1. - ((s.f*theta_der)**2)/6.))
        pre_tot = ((rho_pf - ((s.m*s.f)**2)*s.U(theta)/3.)
                   / (1. - ((s.f*theta_der)**2)/6.))
    else:
        rho_tot = rho_pf
        pre_tot = pre_pf

    return rho_tot, pre_tot


def initialize(species: list[lyas.Species]) -> dict:

    # Construct separate lists of the perfect fluids and other species
    pf_species = []
    scf_species = []
    for s in species:
        if isinstance(s, lyas.PerfectFluid):
            pf_species.append(s)
        elif isinstance(s, lyas.ScalarFieldDarkMatter):
            scf_species.append(s)
        else:
            raise ValueError("Unknown species.")

    # Create a flag which tells whether scf exists or not
    if len(scf_species) == 0:
        has_scf = False
    elif len(scf_species) == 1:
        has_scf = True
    else:
        raise NotImplementedError("More than one scalar field species is not"
                                  " yet implemented.")

    # Create a dictionary for dynamical indexing, and initially
    # populate them with the conformal time and physical time which
    # will be always present
    ind = {
        'tau': 0,
        't': 1
    }
    i = 2  # The integer which denotes the index

    # Below we add elements to this dictionary for additional
    # parameter that we need in the background vector. We should also
    # increase the index.
    if has_scf:
        i += 1
        ind['scf_theta'] = i
        i += 1
        ind['scf_theta_der'] = i

    # Return the useful information as a dictionary
    return {
        'pf_species': pf_species,
        'scf_species': scf_species,
        'has_scf': has_scf,
        'ind': ind
    }


def create_ics(bg: dict, z_ini: float):
    # Get the initial ln_a
    lna_ini = np.log((1. + z_ini)**-1)

    # The initial background vector
    y0 = np.zeros(len(bg['ind'].keys()))

    # We need the initial values of the scalar field if it exists.
    if bg['has_scf']:
        theta_ini, theta_der_ini = bg['scf_species'][0].get_precise_ics(z_ini)
        y0[bg['ind']['scf_theta']] = theta_ini
        y0[bg['ind']['scf_theta_der']] = theta_der_ini

    # Get the initial total energy density
    rho_tot_ini = get_total_densities(bg, lna_ini, y0)[0]

    # Get the initial Hubble
    H_ini = get_hubble(rho_tot_ini)

    # Get the initial conditions for the conformal time tau, and
    # physical time t.
    y0[bg['ind']['tau']] = np.exp(-lna_ini)/H_ini
    y0[bg['ind']['t']] = 0.5/H_ini

    return y0


def integrate(bg: dict, z_ini: float = 1e14, z_fin: float = 0.,
              m_over_H_fin: float | None = None,
              **solve_ivp_args):
    # Get the initial ln_a
    lna_ini = np.log((1. + z_ini)**-1)

    y0 = create_ics(bg, z_ini)

    i = bg['ind']

    # Define the RHS of the system, i.e. the derivatives
    def get_rhs(lna, y):
        rhs = np.zeros(len(y0))
        rho_tot, p_tot = get_total_densities(bg, lna, y)
        H = get_hubble(rho_tot)
        rhs[i['tau']] = np.exp(-lna)/H
        rhs[i['t']] = 1./H

        if bg['has_scf']:
            w = p_tot/rho_tot
            s = bg['scf_species'][0]
            rhs[i['scf_theta']] = y[i['scf_theta_der']]
            rhs[i['scf_theta_der']] = (1.5*(1. - w)*y[i['scf_theta_der']]
                                       - ((s.m)/H)**2)*s.U(y[i['scf_theta']])

        return rhs

    def m_over_H(lna, y):
        rho_tot, p_tot = get_total_densities(bg, lna, y)
        H = get_hubble(rho_tot)
        return np.log(bg['scf_species'][0].m/H) - np.log(m_over_H_fin)
    m_over_H.terminal = True

    if m_over_H_fin is not None:
        events = m_over_H
    else:
        events = None

    # Integrate
    sol = solve_ivp(get_rhs, [lna_ini, np.log(1./(1. + z_fin))],
                    y0=y0, events=events, **solve_ivp_args)

    if sol.success:
        return sol
    else:
        raise RuntimeError(f"Integration failed: {sol.message}.")


def finalize(bg: dict, sol: OdeSolution):
    # Energy and pressure densities of the perfect fluids
    bg['rho'] = {}
    bg['pre'] = {}
    for s in bg['pf_species']:
        bg['rho'][s.index] = s.get_bg_energy_density(sol.t)
        bg['pre'][s.index] = s.get_bg_pressure_density(sol.t)

    return bg
