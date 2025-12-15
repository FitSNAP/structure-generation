from GRSlib.parallel_tools import ParallelTools
from GRSlib.io.input import Config
from GRSlib.converters.convert_factory import convert
from GRSlib.motion.scoring_factory import scoring
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.motion import Gradient, Optimize, Create

import random, copy, os, glob, shutil
import numpy as np
import scipy as sp
from scipy.stats import wasserstein_distance

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

        #Set up the generic variables used everwhere
        self.data = self.config.sections['TARGET'].start_fname
        desc_types = ["current", "target", "prior"] 
        self.descriptors = {key: None for key in desc_types}

        #Set up the super functions 
        self.convert = convert(self.config.sections['BASIS'].descriptor,self.pt,self.config)
        self.loss_func = scoring(self.config.sections['SCORING'].score_type, self.pt, self.config) # Find out which class to call for loss function
#       Instantiate other backbone attributes.
#       self.basis = basis(self.config.sections["BASIS"].descriptor, self.pt, self.config) if "BASIS" in self.config.sections else None

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
        protected = ("pt", "comm")
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
        #print("Called Convert To Descriptors for %s" % data)
        descriptors = self.convert.run_lammps_single(data)
#        if self.config.sections['OUTPUT'].verbosity:
        if not os.path.exists('%s.npy'%data):
            np.save('%s.npy'%data.removesuffix(".data"), descriptors)
            
        return descriptors

    def set_prior(self,structs):
        """
        Accepts a structure (xyz) as input and will return descriptors (D) to be stored and used in conjunction with
        the target, optionally will convert between file types (xyz=lammps-data, ase.Atoms, etc). This is available 
        when a single structure cannot reproduce the target.
        """
        try:
            self.descriptors['prior'] = np.load('prior.npy')
        except:
            pass
        for data in structs:
            prior_desc = self.convert.run_lammps_single(data)
            try:
                self.descriptors['prior'] = np.r_[self.descriptors['prior'], prior_desc] #appends arrays along the first axis (row-wise)
            except:
                self.descriptors['prior'] = prior_desc
        np.save('prior.npy', self.descriptors['prior'])
        
    def update_start(self,data,str_option):
        """
        Used to update the starting structure, usually after a genetic/gradient move has been applied.
        """
        #Save the last structure in a meaningful way, check if other data is present already
        store_id = len(glob.glob(self.config.sections['TARGET'].job_prefix+"*.data"))
        if store_id==0:
            data = self.config.sections['TARGET'].start_fname
        shutil.copyfile(data, self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id)

        if str_option == "Continue":
            #Take the last state from either genetic/gradient move and continue with next move
            data = self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id
        elif str_option == "MinScore":
            #If the score has decreased, continue. Else, retry last structure.
            if(self.before_score >= self.after_score):
                data = self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id
            else:
                try:
                    data = self.config.sections['TARGET'].job_prefix + "_%s.data"%(store_id-1)
                except:
                    data = self.config.sections['TARGET'].job_prefix + "_last.data"                  
        elif str_option == "MaxScore":
            #If the score has increased, continue. Else, retry last structure.
            if(self.before_score < self.after_score):
                data = self.config.sections['TARGET'].job_prefix + "_%s.data"%store_id
            else:
                try:
                    data = self.config.sections['TARGET'].job_prefix + "_%s.data"%(store_id-1)
                except:
                    data = self.config.sections['TARGET'].job_prefix + "_last.data"
#        elif str_option == "Template":
#            #Call structure builder from template, see James' old code
#        elif str_option == "Random":
#            #Call structure buider that creates a random cell of atoms
        elif str_option == "Reset":
            #Fallback to the original input strcture
            data = self.config.sections['TARGET'].start_fname
        elif str_option == "Last":
            #Fallback to the last structure before the most recent genetic/gradient move.
            data = self.config.sections['TARGET'].job_prefix + "_last.data"
        else:
            print("You did not specify a continuation condition (Continue, MinScore, MaxScore, Template, Random, Reset, Last), exiting")
            exit()
        return data

    def get_score(self,data):
        """
        Accepts a structure (xyz) as input and will return descriptors (D), optionally will convert
        between file types (xyz=lammps-data, ase.Atoms, etc)
        """
        #Pass data to, and do something with the functs of scoring
        try:
            self.descriptors['target'] = np.load(self.config.sections['TARGET'].target_fdesc)    
            print("Provided target descriptors superceed target data file")
        except:
            self.descriptors['target'] = self.convert_to_desc(self.config.sections['TARGET'].target_fname)
        self.descriptors['current'] = self.convert_to_desc(data)
        try: # If prior is empty when getting a score, set to starting structure. 
            if self.descriptors.get('prior',None)==None: 
                self.set_prior([self.config.sections['TARGET'].start_fname])
        except:
            pass

        if (np.shape(self.descriptors['current'][1])==np.shape(self.descriptors['target'][1])):
            #Define scoring method now that descriptors and starting data are available
            self.score = Scoring(self.pt, self.config, self.loss_func, self.descriptors) 
            score = self.score.get_score(data) 
        else:
            raise RuntimeError(">>> Found unmatched BASIS for target and current descriptors")
            
        return score

    def get_ensemble_score(self,style):
        #Will score the set of structures stored in self.descriptors["prior"] against those in
        #self.descriptors["target"] using the full distributions of descriptors. A few varients
        #will be reported back, Wasserstein distance and Exact Difference for now.
        if style == 'normalized':
            #Start by normalizing the descriptor histograms (subtract mean, divide by stdev in each dimension)
            target_means = np.average(self.descriptors["target"], axis=0)
            target_stdev = np.std(self.descriptors["target"], axis=0)
            if target_stdev.any() < 1E-6:
                print("WARNING - your target has dimensions of high symmetry and results in an abnormal normalization when ensemble scoring")
            num_descriptors = np.shape(self.descriptors['target'])[1]
            simple_diff = 0.0
            earthmover = 0.0
            normed_target = (self.descriptors["target"]-target_means)/target_stdev
            normed_prior = (self.descriptors["prior"]-target_means)/target_stdev
            for dimension in range(num_descriptors):
                low_limit = min(np.min(normed_target[:,dimension], axis=0),np.min(normed_prior[:,dimension],axis=0))
                high_limit = max(np.max(normed_target[:,dimension], axis=0),np.max(normed_prior[:,dimension],axis=0))
                histo_target, edges = np.histogram(normed_target[:,dimension], bins=100, range=(low_limit,high_limit), density=True)
                histo_prior, edges = np.histogram(normed_prior[:,dimension], bins=100, range=(low_limit,high_limit), density=True)               
                simple_diff += sum(map(abs, (histo_target-histo_prior)))
                earthmover += wasserstein_distance(edges[:-1], edges[:-1], histo_target, histo_prior)
            score = ["Simple Difference",simple_diff/num_descriptors]
            score += (["Wasserstein",earthmover/num_descriptors])

        elif style == 'native':
            num_descriptors = np.shape(self.descriptors['target'])[1]
            simple_diff = 0.0
            earthmover = 0.0
            for dimension in range(num_descriptors):
                low_limit = min(np.min(self.descriptors["target"][:,dimension], axis=0),np.min(self.descriptors["prior"][:,dimension],axis=0))
                high_limit = max(np.max(self.descriptors["target"][:,dimension], axis=0),np.max(self.descriptors["prior"][:,dimension],axis=0))
                histo_target, edges = np.histogram(self.descriptors["target"][:,dimension], bins=100, range=(low_limit,high_limit), density=True)
                histo_prior, edges = np.histogram(self.descriptors["prior"][:,dimension], bins=100, range=(low_limit,high_limit), density=True)
                simple_diff += sum(map(abs, (histo_target-histo_prior)))
                earthmover += wasserstein_distance(edges[:-1], edges[:-1], histo_target, histo_prior)
            score = ["Simple Difference",simple_diff/num_descriptors]
            score += (["Wasserstein",earthmover/num_descriptors])

        return score
    
    def propose_structure(self):
        """
        Currently not usable.
        Propose new structure from random, ase, or templates.
        """
        trial_struct = bulk(self.config.sections["BASIS"].elements[item])
        write('proposed.lammps-data', trial_struct, format='lammps-data', masses=True)
        return 'proposed.lammps-data'

#    @self.pt.single_timeit 
    def genetic_move(self,data):
        """
        Hybridize or mutate a structure using a set of moves sampled via a genetic algorithm
        """
        #1) Propose a set of structures from templates, ase, or random (or read in a list of ase.Aatoms objects)
        #2) Score each of the candidates (ase_to_lammps -> run_single)
        #3) Hybridize, Mutate based on set of rules and probabilities
        #4) Store socring information with best-of-generation and best-overall isolated
        #5) Loop until generation limit or scoring residual below threshold
        if data == None:
            data = self.propose_structure()
        self.descriptors['current']= self.convert_to_desc(data)
        try:
            self.descriptors['target'] = np.load(self.config.sections['TARGET'].target_fdesc)    
        except:
            self.descriptors['target'] = self.convert_to_desc(self.config.sections['TARGET'].target_fname)
   
        self.score = Scoring(self.pt, self.config, self.loss_func, self.descriptors)  # Set scoring class to assign scores to moves
        self.genmove = Optimize(self.pt, self.config, self.score, self.convert) #Set desired motion class with scoring attached
        
        gen_scores = self.genmove.advance_generations(data)
        #self.write_output()
        best_candidate = sorted(gen_scores,key=lambda x: x[3])[0][2]
        return gen_scores, best_candidate


#    @self.pt.single_timeit 
    def gradient_move(self,data):
        """
        Accepts a structure (xyz, ase.Atoms) as input and will return updated structure (xyz, ase.Atoms) that 
        has been modified by motion of atoms on the loss function potential
        """
        #1) Take in target descriptors, convert to moments of descriptor distribution or other loss function
        #2) Take in current descriptors, convert to moments of descriptor distribution or other loss function
        #3) Construct a fictitious potential energy surface based on difference in moments or other loss function (self.loss_func)
        #4) Assemble a LAMMPS input script that overlaps potentials and runs dynamics (self.score)
        #5) Return an updated structure and scoring on the difference in moments or other loss function (self.gradmove)
#        print("Called Gradient_Move")
        
        if data == None:
            data = self.propose_structure()
        self.descriptors['current']= self.convert_to_desc(data)
        try:
            self.descriptors['target'] = np.load(self.config.sections['TARGET'].target_fdesc)    
        except:
            self.descriptors['target'] = self.convert_to_desc(self.config.sections['TARGET'].target_fname)
   
        self.score = Scoring(self.pt, self.config, self.loss_func, self.descriptors)  # Set scoring class to assign scores to moves
        self.gradmove = Gradient(self.pt, self.config, self.score) #Set desired motion class with scoring attached
        grad_type = self.config.sections['GRADIENT'].min_type + '_min'
        event = getattr(self.gradmove, grad_type)
        self.before_score, self.after_score, data = event(data)
        
        store_id = len(glob.glob(self.config.sections['TARGET'].job_prefix+"*.data"))
        with open("scoring_%s.txt"%self.config.sections['TARGET'].job_prefix, "a") as f:
            print(store_id,self.before_score, self.after_score, file=f)
        
        return data

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
        #Can modfiy to take in an output style and call a specific function, for now it is generic and redundant naming
#        @self.pt.single_timeit
        def text_output():
            store_id = len(glob.glob(self.config.sections['TARGET'].job_prefix+"*.data"))
            with open("scoring_%s.txt"%self.config.sections['TARGET'].job_prefix, "a") as f:
                print(store_id,self.before_score, self.after_score, file=f)
        text_output()
