from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.input import Config
from GRSlib.converters.convert_factory import convert
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.motion import Gradient, Genetic

import random
import numpy as np

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
        self.target_desc = []
        self.current_desc = []

#       Instantiate other backbone attributes.
#       self.basis = basis(self.config.sections["BASIS"].descriptor, self.pt, self.config) if "BASIS" in self.config.sections else None

        # Check LAMMPS version if using nonlinear solvers.
        if (hasattr(self.pt, "lammps_version")):
            if (self.pt.lammps_version < 20220915):
                raise Exception(f"Please upgrade LAMMPS to 2022-09-15 or later to use MLIAP based structure searching.")

        #Convert initial target structure if available
        if self.config.sections['TARGET'].target_fname is None:
            print('Target structure not found or undefined')
        else:
            self.target_desc = self.convert_to_desc(self.config.sections['TARGET'].target_fname)

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

    def convert_to_desc(self,data):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        #Pass data to, and do something with the functs of convert
        print("Called Convert To Descriptors for %s" % data)
        self.convert = convert(self.config.sections['BASIS'].descriptor,self.pt,self.config) 
        descriptors = self.convert.run_lammps_single(data)
            
        return descriptors

    def get_score(self,data):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        #Pass data to, and do something with the functs of scoring
        self.current_desc = self.convert_to_desc(data)
        if (np.shape(self.current_desc)==np.shape(self.target_desc)):
            print("Called Scoring Function")
        else:
            raise RuntimeError(">>> Found unmatched for target and current descriptors")

        self.score = Scoring(data, self.current_desc, self.target_desc, self.pt, self.config) 
        score = self.score.get_score()
            
        return score

    def propose_structure(self):
        """
        Propose new structure from random, ase, or templates.
        """
        @self.pt.single_timeit
        def propose_structure(index, desired_size, lattice_type, shape_string, desired_comps={'Zn': 0.5, 'O': 0.5}):
            #1)
             """
        Propose new structure from random, ase, or templates.
        """
        print("Called Propose_Structure")
    
        chems = list(desired_comps.keys())
        vol_base = None
        a_simp = None
    
        if lattice_type == 'wurtzite':
            a_simp = 3.25  
            c_simp = 5.20 
            atoms_base = bulk('ZnO', 'wurtzite', a=a_simp, c=c_simp)  
            vol_base = np.dot(np.cross(atoms_base.get_cell()[0], atoms_base.get_cell()[1]), atoms_base.get_cell()[2])
        else:
            a_simp = 3.0
            atoms_base = bulk(chems[0], lattice_type, a=a_simp)
            vol_base = np.dot(np.cross(atoms_base.get_cell()[0], atoms_base.get_cell()[1]), atoms_base.get_cell()[2])
            a_simp = vol_base**(1/3)
            
        cell_multiples = [combo for combo in itertools.combinations_with_replacement(range(1, 6), 3)]
        structure_multiples = []

        for combo in cell_multiples:
            if shape_string == 'primitive':
                bulk_struct = bulk(chems[0], crystalstructure=lattice_type, a=a_simp) * combo
            elif shape_string == 'cubic':
                bulk_struct = bulk(chems[0], crystalstructure=lattice_type, cubic=True, a=a_simp) * combo
            elif shape_string == 'orthorhombic':
                bulk_struct = bulk(chems[0], crystalstructure=lattice_type, orthorhombic=True, a=a_simp) * combo
            elif shape_string == 'hexagonal':
                bulk_struct = bulk('ZnO', 'wurtzite', a=a_simp, c=c_simp) * combo
            else:
                raise ValueError("shape_string %s is not valid. Please choose: primitive, cubic, orthorhombic, or hexagonal.")
            
            structure_multiples.append(bulk_struct)

        system_size = [len(i) for i in structure_multiples]
        if desired_size in system_size:
            index = system_size.index(desired_size)
            print(index)
            print(structure_multiples[index]) 
            return structure_multiples[index]
        else:
            closest_size = min((size for size in system_size if size >= desired_size), default=None)
            if closest_size is not None:
                index = system_size.index(closest_size)
                print('Desired size not found, Closest:', closest_size)
                print('Structure:', structure_multiples[index])
                return structure_multiples[index]
            else:
                closest_size_l = max((size for size in system_size if size < desired_size), default=None)
                if closest_size_l is not None:
                    index = system_size.index(closest_size_l)
                    print("No exact match and no larger size. Closest:", closest_size_l)
                    print("Structure:", structure_multiples[index])
                    return structure_multiples[index]
                return None
                propose_structure()

    def genetic_move(self,data):
        """
        Hybridize or mutate a structure using a set of moves sampled via a genetic algorithm
        """
#        @self.pt.single_timeit
#        def genetic_move():
#        genetic_move()
        #1) Propose a set of structures from templates, ase, or random (or read in a list of ase.Aatoms objects)
        #2) Score each of the candidates (ase_to_lammps -> run_single)
        #3) Hybridize, Mutate based on set of rules and probabilities
        #4) Store socring information with best-of-generation and best-overall isolated
        #5) Loop until generation limit or scoring residual below threshold
        print("Called Genetic_Move")

        if data == None:
            data = self.propose_structure()

        self.current_desc = self.convert_to_desc(data) 
        self.genmove = Genetic(data, self.current_desc, self.target_desc, self.pt, self.config) 
        self.genmove.something_here()

    def gradient_move(self,data):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
#        @self.pt.single_timeit 
#        def gradient_move():
#        gradient_move()
        #1) Take in target descriptors, convert to moments of descriptor distribution
        #2) Take in current descriptors, convert to moments of descriptor distribution
        #3) Construct a fictitious potential energy surface based on difference in moments
        #4) Assemble a LAMMPS input script that overlaps potentials and runs dynamics
        #5) Return an updated structure and scoring on the difference in moments
        print("Called Gradient_Move")
        
        if data == None:
            data = self.propose_structure()
        
        self.current_desc = self.convert_to_desc(data)
        self.gradmove = Gradient(data, self.current_desc, self.target_desc, self.pt, self.config) 
        if self.config.sections['MOTION'].min_type == 'fire':
            before_score, after_score = self.gradmove.fire_min()
            print(before_score, after_score)
        elif self.config.sections['MOTION'].min_type == 'line':
            before_score, after_score = self.gradmove.line_min()
            print(before_score, after_score)
        elif self.config.sections['MOTION'].min_type == 'box':
            before_score, after_score = self.gradmove.box_min()
            print(before_score, after_score)
        elif self.config.sections['MOTION'].min_type == 'temp':
            before_score, after_score = self.gradmove.run_then_min()
            print(before_score, after_score)

    def baseline_training(self):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
        @self.pt.single_timeit
        def baseline_training():
            #This is more of a 'super' function because it will call many other routines to give the result of a
            #baseline training set with many (hundreds? thousands?) candidate structures. Should return a score
            #of the training diversity based on moments of the descriptor distribution.
            print("Called Baseline_Training")
        baseline_training()

    def write_output(self):
        @self.pt.single_timeit
        def write_output():
            print("Doing some output now")
            #self.output.write_lammps(self.solver.fit)
            #self.output.write_errors(self.solver.errors)
        write_output()