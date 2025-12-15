import numpy as np
#Atoms('',pbc=True)from ase.build import bulk
#from ase.io import read,write
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator, CellBounds
#from ase.ga.startgenerator import StartGenerator
from ase import Atoms,Atom
#from GRSlib.motion.motion import Gradient
from GRSlib.motion.create_helper.ase_tools import ASETools
from GRSlib.motion.genetic_moves.moves import GenMoves
#from ase.data import atomic_numbers
from collections import Counter
import random

class Genetic:
#TODO need to make heavy use of converting between ASE and LAMMPS-Data objects
#TODO each function call here needs to end with scoring call

    def __init__(self, pt, config, convert, scoring, gradmove):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.convert = convert
        self.scoring = scoring
        self.gradmove = gradmove #Set desired motion class with scoring attached

    def crossover(self, parent1, parent2):
        # 1) Takes in a pair of structures, tourny selection should give the most recent winner
        #    and a (random?) second structure from the winners circle or runner up.
        # 2) Crossover will be the 'merger' of these two structures, which for now is the cell from last winner and 
        #    a spliced together set of atoms based on some random dividing line in the atom ids. 
        # 3) Check the spliced cell for accuracy in chemical composition, flip_atoms until close to desired
        # 4) Now generate the remaining population_size - 2  structures as perturbations of the spliced cell. 
        # 5) Return the population 
        crossover_population = []
        crossover_population.append(parent1) #Make sure the two parents make it into the next generation for comparison
        crossover_population.append(parent2) #Make sure the two parents make it into the next generation for comparison
        
        parent1_natom = len(parent1.get_atomic_numbers())
        parent2_natom = len(parent2.get_atomic_numbers())
        for candidate in range(round((self.config.sections["GENETIC"].population_size - 2)/2)): #Populate the remaining with crossovers
            cross_point = np.random.randint(1, parent1_natom-1)
            if (parent1_natom - cross_point) > parent2_natom:
                cross_point = parent1_natom-parent2_natom
                child1 = parent1[:cross_point] + parent2[parent2_natom:]
                child2 = parent1[cross_point:] + parent2[parent2_natom:]
            else:
                child1 = parent1[:cross_point] + parent2[cross_point:]
                child2 = parent2[:cross_point] + parent1[cross_point:]

            pre_move_lammps = self.convert.ase_to_lammps(child1,'tmp')
            grad_type = self.config.sections['GRADIENT'].min_type + '_min'
            event = getattr(self.gradmove, grad_type)
            before_score, after_score, post_move_lammps = event(pre_move_lammps)
            child1 = self.convert.lammps_to_ase(post_move_lammps)

            pre_move_lammps = self.convert.ase_to_lammps(child2,'tmp')
            grad_type = self.config.sections['GRADIENT'].min_type + '_min'
            event = getattr(self.gradmove, grad_type)
            before_score, after_score, post_move_lammps = event(pre_move_lammps)
            child2 = self.convert.lammps_to_ase(post_move_lammps)

            #TODO Need to think if crossovers should impose composition changes, or wait till mutation rounds 
            #chem_comp = child1.get_chemical_formula(mode='all')
            #elements = Counter(chem_comp).keys() #same as set(chem_comp)
            #ele_counts = Counter(chem_comp).items()/len(atoms.numbers()) #counts per unique element

            crossover_population.append(child1)
            crossover_population.append(child2)
        if len(crossover_population) > self.config.sections["GENETIC"].population_size:
            crossover_population.pop(len(crossover_population))
        return crossover_population

    def crossover_ASE(self, parent1, parent2):
        # 1) Takes in a pair of structures, pairs of parents should be the preferred method.
        # 2) Crossover will be the 'merger' of these two structures, which for now is the cell from last winner and 
        #    a spliced together set of atoms based on some random dividing line in the atom ids. 
        # 3) Check the spliced cell for accuracy in chemical composition, flip_atoms until close to desired
        # 4) Return the pair of children from repeating this process once more
        
        atomic_numbers = list(parent1.get_atomic_numbers()) + list(parent2.get_atomic_numbers())
        blmin = closest_distances_generator(atomic_numbers, 0.5)
        slab = Atoms('',pbc=True)
        cellbounds = CellBounds(
            bounds={
                'phi': [20, 160],
                'chi': [20, 160],
                'psi': [20, 160],
                'a': [2, 60],
                'b': [2, 60],
                'c': [2, 60],
            }
        )
        csp = CutAndSplicePairing(
            slab=slab,
            n_top=max([len(parent1),len(parent2)]),
            blmin=blmin,
            p1=1.0,
            p2=0.0,
            minfrac=0.15,
            number_of_variable_cell_vectors=3,
            cellbounds=cellbounds,
            use_tags=False,
        )
        child1 = csp.cross(parent1,parent2)
        pre_move_lammps = self.convert.ase_to_lammps(child1,'child1')
        #Optional minimization after crossover/mutation here, set to none so we can just get the score out
        event = getattr(self.gradmove, 'none_min')
        before_score, child1_score, child1 = event(pre_move_lammps)

        child2 = csp.cross(parent1,parent2)
        pre_move_lammps = self.convert.ase_to_lammps(child1,'child2')
        #Optional minimization after crossover/mutation here, set to none so we can just get the score out
        event = getattr(self.gradmove, 'none_min')
        before_score, child2_score, child2 = event(pre_move_lammps)

        return child1, child1_score, child2, child2_score

    def mutation(self, parent1, parent2, single):
        # 1) Takes in a pair of parent structures
        # 2) Generate a pair of child structures as perturbations of the given cell. 
        # 3) Return the children 
        mutation_options = self.config.sections["GENETIC"].mutation_types
        if not single:
            mutation_array = random.choices(list(mutation_options.keys()), weights=mutation_options.values(), k=2) #unique mutation per parent
        else:
            mutation = random.choices(list(mutation_options.keys()), weights=mutation_options.values(), k=1) #same mutation per parent
            mutation_array = [mutation, mutation]
        event = getattr(GenMoves, mutation_array[0][0])
        child1 = event(parent1,self.config)
        event = getattr(GenMoves, mutation_array[1][0])
        child2 = event(parent2,self.config)
        
        pre_move_lammps = self.convert.ase_to_lammps(child1,'child1')
        #Optional minimization after crossover/mutation here, set to none so we can just get the score out
        event = getattr(self.gradmove, 'none_min')
        before_score, child1_score, child1 = event(pre_move_lammps)
    
        pre_move_lammps = self.convert.ase_to_lammps(child2,'child2')
        #Optional minimization after crossover/mutation here, set to none so we can just get the score out
        event = getattr(self.gradmove, 'none_min')
        before_score, child2_score, child2 = event(pre_move_lammps)

        return child1, child1_score, child2, child2_score
