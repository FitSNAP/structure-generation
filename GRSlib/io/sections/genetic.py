from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Genetic(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['mutation_rate', 'mutation_types', 'population_size', 'ngenerations', 'start_type',
                            'max_atoms', 'min_atoms', 'max_length_aspect', 'max_angle_aspect', 'density_ratio',
                            'composition_constraint', 'composition', 'start_type', 'lattice_type', 'structure_template', 
                            'frac_pop_per_ref', 'dev_per_ref','reference_phases', 'change_ele_tol', 'tournament_size',
                            'replacements', 'multi_mutate','randomseed']
        self._check_section()
        self.mutation_rate = self.get_value("GENETIC", "mutation_rate", 0.5, interpreter="float")
        #self.reference_phases = self.get_value("GENETIC", "reference_phases", "bcc").split()
        self.mutation_types = self.get_value("GENETIC", "mutation_types", {"perturb": 0.5, "change_ele": 0.0, "atom_count" : 0.1, "volume" : 0.2, "minimize" : 0.2}, interpreter="dict")
        self.population_size = self.get_value("GENETIC", "population_size", 20, interpreter="int")
        self.ngenerations = self.get_value("GENETIC", "ngenerations", 10, interpreter="int")
        self.tournament_size = self.get_value("GENETIC", "tournament_size", 10, interpreter="int")
        self.replacements = self.get_value("GENETIC", "replacements", False)
        self.multi_mutate = self.get_value("GENETIC", "multi_mutate", True)
        self.randomseed = self.get_value("GENETIC", "randomseed", 12345, interpreter="int")
        self.max_atoms = self.get_value("GENETIC", "max_atoms", 100, interpreter="int")
        self.change_ele_tol = self.get_value("GENETIC", "change_ele_tol", 0.1, interpreter="float")
        self.min_atoms = self.get_value("GENETIC", "min_atoms", 10, interpreter="int")
        #self.reference_phases = self.get_value("GENETIC","reference_phases", "bcc").split()
        self.max_length_aspect = self.get_value("GENETIC", "max_length_aspec", 3.0, interpreter="float")
        self.max_angle_aspect = self.get_value("GENETIC", "max_angle_aspec", 3.0, interpreter="float")
        self.density_ratio = self.get_value("GENETIC", "density_ratio", 1.3, interpreter="float") #This will allow for 30% changes in either direction of density
        self.composition_constraint = self.get_value("GENETIC", "composition_constraint", None, interpreter="dict")
        if self.composition_constraint is not None:
            self.composition = self.get_value("GENETIC", "composition",{}) #expects dictionary i.e. {'W':0.5, "Re":0.5} 
        
        self.start_type = self.get_value("GENETIC", "start_type", "random", interpreter="str")
        if self.start_type == "random":
            pass # No additional input needed from the user
        elif self.start_type == "lattice":
            self.lattice_type = self.get_value("GENETIC", "lattice_type", None)
        elif self.start_type == "template":
            self.structure_template = self.get_value("GENETIC", "structure_template", None)
        elif self.start_type == "phases":
            self.reference_phases = self.get_value("GENETIC","reference_phases", "bcc").split()
            self.frac_pop_per_ref = [float(k) for k in self.get_value("GENETIC", "frac_pop_per_ref", "1.0").split()]
            self.dev_per_ref =[float(kk) for kk in self.get_value("GENETIC", "dev_per_ref", "0.2").split()]

        self.delete()
