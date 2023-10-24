"""Module for creating the necessary files for a simulation."""
import configparser
from pathlib import Path
import argparse
import os


class LyaSimulation:
    def __init__(self, configfile: str | Path, outputdir: str | Path):
        self.configfile = Path(os.path.expanduser(configfile))
        self.outputdir = Path(os.path.expanduser(outputdir))
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

    def read_genic(self):
        c = self.config['genic']
        self.genic = {
            # Required parameters
            'OutputDir': c.get('OutputDir', 'output'),
            'FileBase': c.get('FileBase', 'IC'),
            'BoxSize': c.getfloat('BoxSize', 10000),
            'Ngrid': c.getint('Ngrid', 128),
            'FileWithInputSpectrum': c.get('FileWithInputSpectrum', 'pk.dat'),
            'Omega0': c.getfloat('Omega0', 0.2814),
            'OmegaBaryon': c.getfloat('OmegaBaryon', 0.0464),
            'OmegaLambda': c.getfloat('OmegaLambda', 0.7186),
            'HubbleParam': c.getfloat('HubbleParam', 0.697),
            'ProduceGas': c.getint('ProduceGas', 0),
            'Redshift': c.getfloat('Redshift', 99),
            'Seed': c.getint('Seed', 256960),
            # Science parameters
            'DifferentTransferFunctions': c.getint('DifferentTransferFunctions', 1),
            'FileWithTransferFunction': c.get('FileWithTransferFunction', 'tk.dat'),
            # Cosmology
            'CMBTemperature': c.getfloat('CMBTemperature', 2.7255),
            'PrimordialAmp': c.getfloat('PrimordialAmp', 2.215e-9),
            'PrimordialIndex': c.getfloat('PrimordialIndex', 0.971),
            # Numerical parameters.
            # TODO: The following needs to be set according to the cluster.
            'MaxMemSizePerNode': c.getfloat('MaxMemSizePerNode', 0.6)
        }

    def read_gadget(self):
        c = self.config['gadget']
        self.gadget = {
            # Required parameters
            'InitCondFile': self.genic['OutputDir'] + self.genic['FileBase'],
            'OutputDir': self.genic['OutputDir'],
            'OutputList': c.get('OutputList', ''),
            # TODO: Update according to the cluster
            'TimeLimitCPU': c.getfloat('TimeLimitCPU', 0),
            'BlackHoleOne': c.getint('BlackHoleOn', 0),
            'StarformationOn': c.getint('StarformationOn', 1),
            'WindOn': c.getint('WindOn', 0),
            # Cosmology
            'TimeMax:': c.getfloat('TimeMax', 0.33333),
            'Omega0': self.genic['Omega0'],
            'CMBTemperature': self.genic['CMBTemperature'],
            # Cooling model parameters
            'CoolingOn': c.getint('CoolingOn', 1),
            'TreeCoolFile': c.get('TreeCoolFile', 'TREECOOL_ep_2018p'),
            
        }

    def write_genic(self, fname: str = 'paramfile.genic'):
        with open(self.outputdir / fname, 'w') as f:
            for key, value in self.genic.items():
                f.write(key + " = " + value.strip("'") + "\n")
        print(f"MP-GenIC config is saved to {self.outputdir / fname}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates necessary files for a lya simulation.")
    parser.add_argument('paramfile')
    parser.add_argument('outputdir')
    args = parser.parse_args()
    sim = LyaSimulation(args.paramfile, args.outputdir)
    sim.read_genic()
    sim.write_genic()
    
