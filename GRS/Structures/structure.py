#!/Users/cmmulle/miniforge3/bin/python
import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from ase.build import bulk
import itertools 
from ase.visualize import view 
import sys

class Structure:
    def __init__(self, index, desired_size, template=None, desired_comps={'Zn': 0.5, 'O': 0.5}, use_template=None, min_typ='temp'):
        self.index = index
        self.template = template
        self.desired_size = desired_size
        self.desired_comps = desired_comps
        self.use_template = use_template
        self.min_typ = min_typ