#!/Users/cmmulle/miniforge3/bin/python
import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from ase.build import bulk
import itertools 
from ase.visualize import view 

#no lattice type for amorphous structures 
#not a bulk struct it will be random
#need to look at what things structure needs

class amorphous(Structure):
    def __init__(self, index, desired_size, template=None, desired_comps={'Zn': 0.5, 'O': 0.5}, use_template=None, min_typ='temp'):
        super().__init__(index, desired_size, template, desired_comps, use_template, min_typ)

    def set_lattice_type(self, lattice_type):
        self.lattice_type = lattice_type

    def set_cell_shape(self, shape_string):
        self.shape_string = shape_string

    def bulk_struct(self):
        assert self.lattice_type is not None, "You must define lattice type before using bulk_struct function."
        chems = list(self.desired_comps.keys())
        vol_base = None
        a_simp = None
        
        if self.lattice_type == 'wurtzite':
            a_simp = 3.25  
            c_simp = 5.20 
            atoms_base = bulk('ZnO', 'wurtzite', a=a_simp, c=c_simp)  
            vol_base = np.dot(np.cross(atoms_base.get_cell()[0], atoms_base.get_cell()[1]), atoms_base.get_cell()[2])
        else:
            atoms_base = bulk(chems[0], self.lattice_type)
            vol_base = np.dot(np.cross(atoms_base.get_cell()[0], atoms_base.get_cell()[1]), atoms_base.get_cell()[2])
            a_simp = vol_base**(1/3)
        cell_multiples = [combo for combo in itertools.combinations_with_replacement(range(1, 6), 3)]
        structure_multiples = []

        for combo in cell_multiples:
            if self.shape_string == 'primitive':
                bulk_struct = bulk(chems[0], crystalstructure=self.lattice_type, a=a_simp) * combo
            elif self.shape_string == 'cubic':
                bulk_struct = bulk(chems[0], crystalstructure=self.lattice_type, cubic=True, a=a_simp) * combo
            elif self.shape_string == 'orthorhombic':
                bulk_struct = bulk(chems[0], crystalstructure=self.lattice_type, orthorhombic=True, a=a_simp) * combo
            elif self.shape_string == 'hexagonal':
                bulk_struct = bulk('ZnO', 'wurtzite', a=a_simp, c=c_simp) * combo
            else:
                raise ValueError("shape_string %s is not valid. Please choose: primitive, cubic, orthorhombic, or hexagonal.")
            structure_multiples.append(bulk_struct)

        system_size = [len(i) for i in structure_multiples]
        if self.desired_size in system_size:
            index = system_size.index(self.desired_size)
            print(index)
            print(structure_multiples[index]) 
            return structure_multiples[index]
        else:
            closest_size = min((size for size in system_size if size >= self.desired_size), default=None)
            if closest_size is not None:
                index = system_size.index(closest_size)
                print('Desired size not found, Closest:', closest_size)
                print('Structure:', structure_multiples[index])
                return structure_multiples[index]
            else:
                closest_size_l = max((size for size in system_size if size < self.desired_size), default=None)
                if closest_size_l is not None:
                    index = system_size.index(closest_size_l)
                    print("No exact match and no larger size. Closest:", closest_size_l)
                    print("Structure:", structure_multiples[index])
                    return structure_multiples[index]
                return None
def test_amorphous():
    amorphous_test1 = amorphous(1,100)