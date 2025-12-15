import ctypes
from GRSlib.converters.convert import Convert
import numpy as np
import copy


class Base(Convert):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {}
        self._i = 0
        self._lmp = None
        self.pt.check_lammps()

#    def process_single(self, data, i=0):
        """
        Calculate descriptors on a single configuration without touching the shared arrays.

        Args:
            data: dictionary of structural and fitting info for a configuration in grs
                  data dictionary format.
            i: integer index which is optional, mainly for debugging purposes.
        
        Returns: 
            - A : matrix of descriptors depending on settings declared in `BASIS`. If 
              `bikflag` is 0 (default) then A has 1 and 0s in the first column since it is ready to 
              fit with linear solvers; the descriptors are also divided by no. atoms in this case. 
              If `bikflag` is 1, then A is simply an unaltered per-atom descriptor matrix.
        """
#        self._data = data
#        self._i = i
#        self._initialize_lammps()
#        self._prepare_lammps()
#        self._run_lammps()
#        a,b,w = self._collect_lammps_single()
#        self._lmp = self.pt.close_lammps()
#        return a,b,w
    
    def _initialize_lammps(self, printlammps=0):
#        self._lmp = self.pt.initialize_lammps(self.config.args.lammpslog, printlammps)
        self._lmp = self.pt.initialize_lammps('log.lammps',0)

    def _set_structure(self):
        self._lmp.command("clear")
        self._lmp.command("units metal")
        self._lmp.command("atom_style atomic")

        lmp_setup = _extract_commands("""
                        atom_modify map array sort 0 2.0
                        box tilt large""")
        for line in lmp_setup:
            self._lmp.command(line)

        #self._set_box() #might not need this if reading from file

    def _set_neighbor_list(self):
        self._lmp.command("mass * 1.0e-20")
        self._lmp.command("neighbor 1.0e-20 nsq")
        self._lmp.command("neigh_modify one 10000")

    def _set_box_helper(self, numtypes):
        self._lmp.command("boundary p p p")
        ((ax, bx, cx),
         (ay, by, cy),
         (az, bz, cz)) = self._data["Lattice"]

        assert all(abs(c) < 1e-10 for c in (ay, az, bz)), \
            "Cell not normalized for lammps!\nGroup and configuration: {} {}".format(self._data["Group"], self._data["File"])
        region_command = \
            f"region pybox prism 0 {ax:20.20g} 0 {by:20.20g} 0 {cz:20.20g} {bx:20.20g} {cx:20.20g} {cy:20.20g}"
        self._lmp.command(region_command)
        self._lmp.command(f"create_box {numtypes} pybox")

    def _run_lammps(self):
        self._lmp.command("run 0")

def _extract_atom_ids(self, num_atoms):
    # helper function to account for change in LAMMPS numpy_wrapper method name
    try:
        ids = self._lmp.numpy.extract_atom(name="id", nelem=num_atoms).ravel()
    except:
        ids = self._lmp.numpy.extract_atom_iarray(name="id", nelem=num_atoms).ravel()
    return ids

def _extract_atom_positions(self, num_atoms):
    # helper function to account for change in LAMMPS numpy_wrapper method name
    try:
        pos = self._lmp.numpy.extract_atom(name="x", nelem=num_atoms, dim=3)
    except:
        pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
    return pos

def _extract_atom_types(self, num_atoms):
    # helper function to account for change in LAMMPS numpy_wrapper method name
    try:
        types = self._lmp.numpy.extract_atom(name="type", nelem=num_atoms).ravel()
    except:
        types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
    return types


def _extract_compute_np(lmp, name, compute_style, result_type, array_shape):
    """
    Convert a lammps compute to a numpy array.
    Assumes the compute stores floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is
    skipped and a python float is returned.
    From LAMMPS/src/library.cpp:
    style = 0 for global data, 1 for per-atom data, 2 for local data
    type = 0 for scalar, 1 for vector, 2 for array
    """

    if array_shape is None:
        array_np = lmp.numpy.extract_compute(name,compute_style, result_type)
        return array_np
    else:
        ptr = lmp.extract_compute(name, compute_style, result_type)
        if result_type == 0:

            # no casting needed, lammps.py already works

            return ptr
        if result_type == 2:
            ptr = ptr.contents
        total_size = np.prod(array_shape)
        buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double * total_size))
        array_np = copy.deepcopy(np.frombuffer(buffer_ptr.contents, dtype=float))
#        array_np = copy.deepcopy(buffer_ptr.contents)
        array_np.shape = array_shape
        reshaped_array_np = np.delete(array_np, np.s_[-1:], axis=1)
        #LAMMPS is returning an extra descriptor at the end of the list, deleting by reshaping
        return reshaped_array_np

def _extract_commands(string):
    return [x for x in string.splitlines() if x.strip() != '']

