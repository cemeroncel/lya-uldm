"""Module for creating the necessary files for a simulation."""
from configobj import ConfigObj
import os
from classy import Class
import numpy as np
import math


class Simulation:
    def __init__(self, *, box: float, ngrid: int, OmegaMatter: float,
                 OmegaBaryon: float, OmegaLambda: float, h: float,
                 output_list: list[float], cpu_time_hr: float,
                 z_start: float = 99., seed: int = 960169,
                 As: float = 2.100549e-9, ns: float = 0.9660499,
                 a_end: float = 1.):
        # Input parameters
        self.box = box
        self.ngrid = ngrid
        self.OmegaMatter = OmegaMatter
        self.OmegaBaryon = OmegaBaryon
        self.OmegaLambda = OmegaLambda
        self.h = h
        self.output_list = output_list
        self.z_start = z_start
        self.seed = seed
        self.As = As
        self.ns = ns
        self.a_end = a_end

        # Hard-coded parameters
        self.output = 'output'
        self.ic = 'IC'
        self.pk_filename = 'class_pk.dat'
        self.tk_filename = 'class_tk.dat'

        # TODO: Parameters that need to be set according to the cluster
        self.max_mem_size_per_node = 0.6
        self.cpu_time_hr = cpu_time_hr

    def write_genic(self):
        config = ConfigObj()
        config.filename = 'paramfile.genic'

        # Required parameters
        config['OutputDir'] = self.output
        config['FileBase'] = self.ic
        config['BoxSize'] = self.box
        config['Ngrid'] = self.ngrid
        config['WhichSpectrum'] = 2
        config['FileWithInputSpectrum'] = self.pk_filename
        config['FileWithTransferFunction'] = self.tk_filename
        config['Omega0'] = self.OmegaMatter
        config['OmegaBaryon'] = self.OmegaBaryon
        config['OmegaLambda'] = self.OmegaLambda
        config['HubbleParam'] = self.h
        config['ProduceGas'] = self.produce_gas
        config['Redshift'] = self.z_start
        config['Seed'] = self.seed
        config['DifferentTransferFunctions'] = self.different_transfer
        config['ScaleDepVelocity'] = self.scale_dep_velocity
        config['PrimordialAmp'] = self.As
        config['PrimordialIndex'] = self.ns
        config['MaxMemSizePerNode'] = self.max_mem_size_per_node

        config.write()

    def write_gadget(self):
        config = ConfigObj()
        config.filename = 'paramfile.gadget'

        # Required parameters
        config['InitCondFile'] = self.output + '/' + self.ic
        config['OutputDir'] = self.output
        config['OutputList'] = str(self.output_list).strip('[]').strip()
        config['TimeLimitCPU'] = 60*60*self.cpu_time_hr
        config['SnapshotWithFOF'] = self.snapshot_with_fof
        config['BlackHoleOn'] = self.black_hole_on
        config['StarformationOn'] = self.star_formation_on
        config['WindOn'] = self.wind_on

        # Cosmology parameters
        config['TimeMax'] = self.a_end
        config['Omega0'] = self.OmegaMatter
        config['OmegaBaryon'] = self.OmegaLambda
        config['HubbleParam'] = self.h
        config['RadiationOn'] = 1

        # Cooling model parameters
        config['CoolingOn'] = self.cooling_on
        if self.cooling_on == 1:
            # TODO: Check if this file exists
            config['TreeCoolFile'] = 'TREECOOL'
        config['HeliumHeatOn'] = self.helium_heat_on
        if self.helium_heat_on == 1:
            config['HeliumHeatThresh'] = self.helium_heat_thresh
            config['HeliumHeatAmp'] = self.helium_heat_amp
            config['HeliumHeatExp'] = self.helium_heat_exp

        # Star formation model parameters
        if self.star_formation_on == 1:
            config['StarformationCriterion'] = self.sf_criterion
            config['QuickLymanAlphaProbability'] = self.quick_lya
            config['CritOverDensity'] = self.crit_over_density

        # Wind (Stellar Feedback Model Parameters)
        config['WindModel'] = 'nowind'

        # Other parameters
        config['DensityIndependentSphOne'] = self.density_independent_sph
        config['HydroOne'] = self.hydro
        config['DensityKernelType'] = self.density_kernel
        config['MaxMemSizePerNode'] = self.max_mem_size_per_node
        config['InitGasTemp'] = self.init_gas_temp
        config['MinGasTemp'] = self.min_gas_temp

        # Write to config
        config.write()

    def make_class_power(self):
        # The box size is given in internal units.
        # Default value for Unit length in MPGadget is 1.0 kpc/h
        box_in_Mpc_over_h = 1e-3*self.box

        self.class_cosmo = Class()
        self.class_cosmo.set({
            'output': 'mPk,dTk,vTk',
            'extra_metric_transfer_functions': 'yes',
            'gauge': 'synchronous',
            'h': self.h,
            'omega_b': (self.h**2)*self.OmegaBaryon,
            'omega_cdm': (self.h**2)*(self.OmegaMatter - self.OmegaBaryon),
            'A_s': self.As,
            'n_s': self.ns,
            'P_k_max_h/Mpc': max(10, 2*math.pi*self.ngrid*4/box_in_Mpc_over_h),
            'z_pk': self.z_start,
            'tol_background_integration': 1e-9,
            # 'tol_perturb_integration': 1e-9,
            'tol_thermo_integration': 1e-5,
            'k_per_decade_for_pk': 50,
            'k_bao_width': 8,
            'k_per_decade_for_bao': 200,
            'neglect_CMB_sources_below_visibility': 1e-30,
            'transfer_neglect_late_source': 3000.,
            'l_max_g': 50,
            'l_max_ur': 150
        })
        self.class_cosmo.compute()

    def class_transfer_save(self):
        transfer = self.class_cosmo.get_transfer(z=self.z_start)
        trans_fname = self.tk_filename
        trans_array = []
        header = (f"Transfer functions T_i(k) for adiabatic (AD) mode"
                  "(normalized to initial curvature=1) at redshift "
                  f"z={self.z_start}\n"
                  f"for k={transfer['k (h/Mpc)'][0]} to "
                  f"{transfer['k (h/Mpc)'][-1]} h/Mpc,\n"
                  f"number of wavenumbers equal to {len(transfer['k (h/Mpc)'])}\n"
                  "d_i   stands for (delta rho_i/rho_i)(k,z) with above normalization\n"
                  "d_tot stands for (delta rho_tot/rho_tot)(k,z) with rho_Lambda NOT included in rho_tot\n"
                  "(note that this differs from the transfer function output from CAMB/CMBFAST, which gives the same\n"
                  "t_i   stands for theta_i(k,z) with above normalization\n"
                  "t_tot stands for (sum_i [rho_i+p_i] theta_i)/(sum_i [rho_i+p_i]))(k,z)\n\n")

        i = 1
        col_number = len(transfer.keys())
        for k, v in transfer.items():
            trans_array.append(v)
            header += (str(i) + ':' +str(k))
            if i < col_number:
                header += '\t'
            i += 1

        np.savetxt(trans_fname, trans_array, header=header, fmt='%.12e',
                   delimiter='\t')


class LyaSimulation(Simulation):
    def __init__(self,
                 helium_heat_thresh: float = 10.,
                 helium_heat_amp: float = 1.,
                 helium_heat_exp: float = 1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.produce_gas = 1
        self.different_transfer = 1
        self.scale_dep_velocity = 1
        self.snapshot_with_fof = 0
        self.black_hole_on = 0
        self.star_formation_on = 1
        self.wind_on = 0
        self.cooling_on = 1
        self.helium_heat_on = 1
        self.helium_heat_thresh = helium_heat_thresh
        self.helium_heat_amp = helium_heat_amp
        self.helium_heat_exp = helium_heat_exp
        self.sf_criterion = 'density'
        self.quick_lya = 1
        self.crit_over_density = 1000
        self.density_independent_sph = 0
        self.hydro = 1
        self.density_kernel = 'cubic'
        self.init_gas_temp = 270.
        self.min_gas_temp = 100.
