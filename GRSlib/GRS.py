from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.input import Config
import random


class GRS:
    """ 
    >Big comment goes here explaining what the code does
    Args:
        input (str): Optional dictionary or path to input file when using library mode; defaults to 
                     None for executable use.
        comm: Optional MPI communicator when using library mode; defaults to None.

    Attributes:
        pt (:obj:`class` ParallelTools): Instance of the ParallelTools class for helping MPI 
                                         communication and shared arrays.
        config (:obj:`class` Config): Instance of the Config class for initializing settings, 
                                      initialized with a ParallelTools instance.
        >Update once more of the code structure is fleshed out
    """
    def __init__(self, input=None, comm=None, arglist: list=[]):
        self.comm = comm
        # Instantiate ParallelTools and Config instances belonging to this GRS instance.
        # NOTE: Each proc in `comm` creates a different `pt` object, but shared arrays still share 
        #       memory within `comm`.
        self.pt = ParallelTools(comm=comm)
        self.pt.all_barrier()
        self.config = Config(self.pt, input, arguments_lst=arglist)

        # Instantiate other backbone attributes.
        self.basis = basis(self.config.sections["BASIS"].descriptor, self.pt, self.config) if "BASIS" in self.config.sections else None
        #self.convert = convert(self.pt,self.config)

        # Check LAMMPS version if using nonlinear solvers.
        if (hasattr(self.pt, "lammps_version")):
            if (self.pt.lammps_version < 20220915):
                raise Exception(f"Please upgrade LAMMPS to 2022-09-15 or later to use MLIAP based structure searching.")
          
    def __del__(self):
        """Override deletion statement to free shared arrays owned by this instance."""
        self.pt.free()
        del self

    def __setattr__(self, name: str, value):
        """
        Override set attribute statement to prevent overwriting important attributes of an instance.
        """
        protected = ("pt", "config")
        if name in protected and hasattr(self, name):
            raise AttributeError(f"Overwriting {name} is not allowed; instead change {name} in place.")
        else:
            super().__setattr__(name, value)

    def convert(self):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        @self.pt.single_timeit
        def convert():
            #Pass data to, and do something with the functs of convert
            convert()
    
    def genetic_move(self):
        """
        Propose new structure, or adapt existing ones using a set of moves sampled via a genetic algorithm
        """
        @self.pt.single_timeit
        def genetic_move():
            #Pass data to, and do something with the functs of genetic_move
            genetic_move()

    def gradient_move(self):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
        @self.pt.single_timeit
        def gradient_move():
            #Pass data to, and do something with the functs of gradient_move
            gradient_move()
    def baseline_training(self):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
        @self.pt.single_timeit
        def baseline_training():
            #Pass data to, and do something with the functs of baseline_training
            baseline_training()

    def write_output(self):
        @self.pt.single_timeit
        def write_output():
            if not self.config.args.perform_fit:
                return
            self.output.output(self.solver.fit, self.solver.errors)

            #self.output.write_lammps(self.solver.fit)
            #self.output.write_errors(self.solver.errors)
        write_output()
