from ase.io import read,write
from ase import Atoms,Atom
from ase.ga.utilities import closest_distances_generator, CellBounds
from ase.ga.startgenerator import StartGenerator
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.neighborlist import primitive_neighbor_list
import numpy as np
import random
from collections import Counter

# Lowest level functions that can be used to modif structures, inherited class not needed since scoring will happen in
# motion/genetic.py. This collection of functions is mostly to avoid clustter and massive files where more abstract 
# things are happening.

class GenMoves():
    
    def atom_count(atoms,config):
        #From the user input of density ratio, figure out the bounds of number of atoms to add/remove
        #Roll dice on the +/- bounds and adjust the ase.atom object that is read in.
        cell = atoms.get_cell() #Carry over the cell size from the supecell
        cell_lenx, cell_leny, cell_lenz = atoms.cell.cellpar()[0],atoms.cell.cellpar()[1],atoms.cell.cellpar()[2]
        natoms = len(atoms.get_atomic_numbers())
        chem_comp = list(atoms.symbols) #atoms.get_chemical_formula(mode='all')
        elements = list(Counter(chem_comp).keys()) #same as set(chem_comp)
        ele_counts = list(Counter(chem_comp).items()) #counts per unique element
        new_positions = atoms.get_positions()
        max_deladd = np.abs(natoms - round(natoms*config.sections["GENETIC"].density_ratio))
        change_count = random.randint(-max_deladd,max_deladd)
        if change_count < 0:
            for i in range(abs(change_count)):
                del atoms[random.randint(0,len(atoms.get_atomic_numbers())-1)] #Needs to be atoms.get_atomic_numbers not natoms
            chem_comp = list(atoms.symbols) #atoms.get_chemical_formula(mode='all')
            elements = list(Counter(chem_comp).keys()) #same as set(chem_comp)
            ele_counts = list(Counter(chem_comp).items()) #counts per unique element
            sym_comp = ""
            for ele in range(len(elements)):
                sym_comp += elements[ele]+str(ele_counts[ele][1])
            new_positions = atoms.get_positions()
            new_atoms = Atoms(symbols=sym_comp,positions=new_positions, cell=cell, pbc=[1,1,1])
        else:
            for i in range(abs(change_count)):
                tmp_x,tmp_y,tmp_z = np.random.uniform(low=0.0,high=cell_lenx),np.random.uniform(low=0.0,high=cell_leny),np.random.uniform(low=0.0,high=cell_lenz)
                new_positions = np.append(new_positions,[[tmp_x,tmp_y,tmp_z]], axis=0)
                chem_comp.append(random.choice(list(set(chem_comp))))
            sym_comp = ""
            elements = list(Counter(chem_comp).keys()) #same as set(chem_comp)
            ele_counts = list(Counter(chem_comp).items()) #counts per unique element
            for ele in range(len(elements)):
                sym_comp += elements[ele]+str(ele_counts[ele][1])
            new_atoms = Atoms(symbols=sym_comp,positions=new_positions, cell=cell, pbc=[1,1,1])
        return new_atoms

    def volume(atoms,config):
        #From the user input of density ratio, take cube root and roll dice for lx,ly,lz,alpha,beta,gamma
        cell = atoms.get_cell() #Carry over the cell size from the supecell
        scaled_positions = atoms.get_scaled_positions()
        atom_symbols = atoms.symbols
        scale_matrix = np.eye(3,dtype=float)*0.5*(config.sections["GENETIC"].density_ratio)**(1./3.)
        new_cell = np.matmul(cell,scale_matrix)
        new_positions = np.matmul(scaled_positions,new_cell)
        new_atoms = Atoms(atom_symbols,positions=new_positions, cell=new_cell, pbc=[1,1,1])
        return new_atoms

    def ortho_cell(atoms,config):
        #Convert the cell to an orthorhombic cell
        cell_lenx, cell_leny, cell_lenz = atoms.cell.cellpar()[0],atoms.cell.cellpar()[1],atoms.cell.cellpar()[2]
        scaled_positions = atoms.get_scaled_positions()
        atom_symbols = atoms.symbols
        new_cell = np.eye(3,dtype=float)
        new_cell[0][0] = atoms.cell.cellpar()[0]
        new_cell[1][1] = atoms.cell.cellpar()[1]
        new_cell[2][2] = atoms.cell.cellpar()[2]
        new_positions = np.matmul(scaled_positions,new_cell)
        new_atoms = Atoms(atom_symbols,positions=new_positions, cell=new_cell, pbc=[1,1,1])
        return new_atoms


    def perturb(atoms,config):
        new_cell = atoms.get_cell() #Carry over the cell size from the supecell
        atom_symbols = atoms.symbols
        new_positions = atoms.get_positions()
        atom_length = (1/(3.)**(1./2.))*(len(atoms.get_atomic_numbers())/atoms.get_volume())*(1./3.) # Linear distance from average atomic volume, becomes max displacement distance
        change_count = random.randint(1,round(len(atoms.get_atomic_numbers())/2)) #Perturb up to one-half the atom positions
        for i in range(change_count):
            pertub_id = random.randint(0,len(atoms.get_atomic_numbers())-1)
            new_positions[pertub_id][0] += np.random.uniform(low=-atom_length,high=atom_length)
            new_positions[pertub_id][1] += np.random.uniform(low=-atom_length,high=atom_length)
            new_positions[pertub_id][2] += np.random.uniform(low=-atom_length,high=atom_length)

        new_atoms = Atoms(atom_symbols,positions=new_positions, cell=new_cell, pbc=[1,1,1])
        return new_atoms

    def change_ele(atoms,config):
        #NOTE this function changes elements correctly, but candidates are not being used correctly in
        # motion.py during tournament selection
        tol = config.sections["GENETIC"].change_ele_tol
        #tol = 0.1 # % tolerance for composition constraint
        tol_int = int(round(tol*len(atoms)))
        pm_frac = tol_int/len(atoms)
        assert pm_frac >= 0, "need to adjust tolerance in change_ele"

        #this splits CaMgCaMgMg into  ['C','a','M','g']
        #chem_comp = atoms.get_chemical_formula(mode='all')
        #elements = list(Counter(chem_comp).keys()) #same as set(chem_comp) 
        #ele_counts = Counter(chem_comp).items()/len(atoms.numbers) #counts per unique element
        uniques = config.sections["BASIS"].elements.copy()
        elements = [atom.symbol for atom in atoms]
        ele_counts_raw = Counter(elements)
        ele_counts_dct = {ue:ele_counts_raw[ue]/len(atoms) for ue in uniques}
        ele_counts = list(ele_counts_dct.values())
        old_comp = ele_counts.copy()
        #min/max # of indices to perturb
        mn = 1/len(atoms)
        mx = (len(atoms)-1)/len(atoms)
        itr = 0
        target_dct = config.sections["GENETIC"].composition_constraint.copy()
        target_comp = tuple(list(target_dct.values()))
        le_cond = any([icomp < target_comp[ii] - pm_frac for ii,icomp in enumerate(ele_counts)])
        ge_cond = any([icomp > target_comp[ii] + pm_frac for ii,icomp in enumerate(ele_counts)])
        new_atoms = atoms.copy()
        while itr == 0 or le_cond or ge_cond:
            fraction = np.random.uniform(mn,mx)
            pert_inds = np.random.choice(range(len(atoms)),size=int(len(atoms)*fraction),replace=False )
            for pert_ind in pert_inds:
                flip_current = new_atoms[pert_ind].symbol
                excluded = [typ for typ in elements if typ != flip_current]
                #NOTE
                #flip_to_type = np.random.choice([ue for ue in uniques if ue != flip_current])
                flip_to_type = np.random.choice([ue for ue in uniques])
                new_atoms[pert_ind].symbol = flip_to_type
            new_elements = [atom.symbol for atom in new_atoms]
            new_ele_counts_raw = Counter(new_elements)
            ele_counts_dct = {ue:new_ele_counts_raw[ue]/len(atoms) for ue in uniques}
            ele_counts = list(ele_counts_dct.values())
            compare_comp = tuple(list(ele_counts_dct.values()))
            le_cond = any([icomp < target_comp[ii] - pm_frac for ii,icomp in enumerate(ele_counts)])
            ge_cond = any([icomp > target_comp[ii] + pm_frac for ii,icomp in enumerate(ele_counts)])
            itr += 1
        new_elements = [atom.symbol for atom in new_atoms]
        new_ele_counts_raw = Counter(new_elements)
        ele_counts_dct = {ue:new_ele_counts_raw[ue]/len(atoms) for ue in uniques}
        ele_counts = list(ele_counts_dct.values())
        le_cond = any([icomp < target_comp[ii] - pm_frac for ii,icomp in enumerate(ele_counts)])
        ge_cond = any([icomp > target_comp[ii] + pm_frac for ii,icomp in enumerate(ele_counts)])
        #print('final',itr,target_comp,ele_counts, le_cond,ge_cond)
        return new_atoms

    def minimize(atoms,config):
        #Do nothing because it will be relaxed upon returning to genetic.py
        return atoms    
