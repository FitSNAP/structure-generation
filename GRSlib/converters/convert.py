from GRSlib.parallel_tools import ParallelTools
from ase.io import read, write, lammpsdata
from ase.data import atomic_masses, atomic_numbers


class Convert:

    def __init__(self, name, pt, config):
        self.pt = pt #ParallelTools()
        self.config = config #Config()

#   Ok technically these functions dont do anything right now, but I'm holding onto them as
#   boilerplate for the ASE conversions, or descriptor conversions that dont need LAMMPS

    def ase_to_lammps(self,data,*args):
        types = self.config.sections["BASIS"].elements
        Z_of_type = {i+1:atomic_numbers[ele] for i,ele in enumerate(types)}
        """
        Takes in an ase.Atoms object and writes a lammps-data, returns the file name
        """
        fname = args[0]
        #write(fname, data, format='lammps-data', masses=True, Z_of_type=Z_of_type)
        write(fname, data, format='lammps-data', masses=False)
        
        return fname

    def lammps_to_ase(self,data):
        types = self.config.sections["BASIS"].elements
        
        Z_of_type = {i+1:atomic_numbers[ele] for i,ele in enumerate(types)}
        """
        Takes in a lammps-data file and returns an ase.Atoms object
        """
        try:
            ase_data = read(data,format='lammps-data',Z_of_type=Z_of_type)
        except:
            ase_data = read(data+".lammps-data",format='lammps-data',Z_of_type=Z_of_type)
        #print('in GRSlib/converters/convert.py lammps_to_ase ',ase_data,types)
        return ase_data

    def lammps_ace(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in the ACE basis set
        """
        descriptor_vals = Ace.run_lammps_single(data)
        return descriptor_vals

    def lammps_snap(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in the SNAP basis set (Not Implemented ATM)
        """
        descriptor_vals = Snap.run_lammps_single(data)
        return descriptor_vals

    def lammps_custom(self,data):
        """
        Takes in an lammps-data file and converts to descriptors in a custom basis set (Not Implemented ATM)
        """
        descriptor_vals = Custom.run_lammps_single(data)
        return descriptor_vals

