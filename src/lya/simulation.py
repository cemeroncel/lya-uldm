"""Module for creating the necessary files for a simulation."""
from configobj import ConfigObj


class Simulation:
    def __init__(self, *, box: float, ngrid: int, OmegaMatter: float,
                 OmegaBaryon: float, OmegaLambda: float, h: float,
                 output_list: list[float], cpu_time_hr: float,
                 z_start: float = 99., seed: int = 960169,
                 As: float = 2.215e-9, ns: float = 0.971,
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
        config['FileBase'] = self.IC
        config['BoxSize'] = self.box
        config['Ngrid'] = self.ngrid
        config['WhichSpectrum'] = 2
        config['FileWithInputSpectrum'] = self.pk_filename
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
        config['InitCondFile'] = self.output + '/' + self.IC
        config['OutputDir'] = self.output
        config['OutputList'] = self.output_list  # Check if this parses correctly.
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
