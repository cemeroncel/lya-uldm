"""Module for creating the necessary files for a simulation."""
from configobj import ConfigObj


class Simulation:
    def __init__(self, *, box: float, ngrid: int, OmegaMatter: float,
                 OmegaBaryon: float, OmegaLambda: float, h: float,
                 output_list: list[float], cpu_time_hr: float,
                 z_start: float = 99., seed: int = 960169,
                 As: float = 2.215e-9, ns: float = 0.971):
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

        config['InitCondFile'] = self.output + '/' + self.IC
        config['OutputDir'] = self.output
        config['OutputList'] = self.output_list  # Check if this parses correctly.
        config['TimeLimitCPU'] = 60*60*self.cpu_time_hr


class LyaSimulation(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.produce_gas = 1
        self.different_transfer = 1
        self.scale_dep_velocity = 1
