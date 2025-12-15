from GRSlib.motion.scoring import Scoring
import jax.numpy as jnp
import numpy as np
import scipy as sp
from scipy.stats import wasserstein_distance
from jax import grad, jit
from functools import partial

class Wasserstein(Scoring):
    def __init__(self, *args): #pt, config, target_desc, prior_desc):
        self.pt, self.config, descriptors = args
        self.target_desc = descriptors.get('target',None).copy() 
        self.prior_desc = descriptors.get('prior',None).copy()
        self.n_descriptors = np.shape(self.target_desc)[1]
        self.mask = list(range(self.n_descriptors))
        self.n_params = 1 #Variables LAMMPS needs to know about
        self.n_elements = self.config.sections['BASIS'].numtypes #Variables LAMMPS needs to know about
        self.loss_ff_grad = grad(self.construct_loss)
        self.grad_loss = grad(self.construct_loss)
        self.mode = "update"

    def __call__(self, *args):
        #The arguments that this function brings in are super improtant and are expected by LAMMPS MLIAP package.
        #They are (elems, current_desc, beta, energy)
        #Integer values of LAMMPS atom types are in elems
        #Descriptors as a per-atom array into current_desc.
        #Per-atom forces are expected for beta
        #Per-atom energy is expected for energy, need to do some testing if per-atom values can be reported back.
        if self.mode=="score":     
            elems, current_desc, beta, energy = args
            self.n_atoms = np.shape(current_desc)[0]

            #TODO Explain
            score = self.construct_loss(current_desc, self.target_desc)
            energy[:] = 0
            energy[0] = self.config.sections["SCORING"].strength_target*score #Scaled score (energy) between current and target
            forces = self.grad_loss(current_desc, self.target_desc) #Forces between current and target
            beta[:,:]= 0
            beta[:,:] = self.config.sections["SCORING"].strength_target*forces #Scaled forces between current and target

            #TODO Explain
            score = self.construct_loss(self.prior_desc, self.target_desc)
#            energy[0] += self.config.sections["SCORING"].strength_prior*score #Scaled score (energy) between current and prior
#            print("     Target, Prior Scores: ", energy[0], score)
            forces = self.grad_loss(current_desc, self.prior_desc) #Forces between current and prior structures
            beta[:,:] += self.config.sections["SCORING"].strength_prior*forces #Scaled forces between current and prior

        elif self.mode=="update":
            self.update(args)
            beta = self.grad_loss(self.target_desc, self.target_desc)


    def set_mode_update(self):
        self.mode="update"

    def set_mode_score(self):
        self.mode="score"

    def update(self,*args):
        pt, config, descriptors = args[0]
        self.target_desc = descriptors.get('target',None).copy()
        self.prior_desc = descriptors.get('prior',None).copy()
        self.n_descriptors = np.shape(self.target_desc)[1]
        if self.config.sections["SCORING"].smartmask > 0:
            indices = []
            target_std = np.std(self.target_desc, axis=0)
            nmax_var = len(target_std) - int(np.ceil(self.config.sections["SCORING"].smartmask/2))
            nmin_var = int(np.floor(self.config.sections["SCORING"].smartmask/2))
            list_max_var = sorted(target_std,key=lambda x: x)[nmax_var:]
            list_min_var = sorted(target_std,key=lambda x: x)[:nmin_var]
            for val in list_max_var:
                indices.append(np.where(target_std==val)[0][0])
            for val in list_min_var:
                indices.append(np.where(target_std==val)[0][0])

            self.mask = np.zeros(len(target_std), dtype=int)
            self.mask[indices] = 1
            
        else:
            self.mask = np.zeros(self.n_descriptors, dtype=int) + 1


        if self.n_elements > 1:
            self.current_desc = self.current_desc.flatten()
            self.target_desc = self.target_desc.flatten()
            self.prior_desc = self.prior_desc.flatten()
        self.mode = "score"

    @partial(jit, static_argnums=(0,))
    def construct_loss(self, current_desc, target_desc):
        #This needs to be a dynamic call like is done in descriptor calcs 
        num_descriptors = jnp.shape(self.target_desc)[1]
        earthmover = 0.0
        for dimension in range(num_descriptors):
            curr_max, curr_min = jnp.amax(current_desc[:,dimension], axis=0),jnp.amin(current_desc[:,dimension], axis=0)
            target_max, target_min = jnp.amax(target_desc[:,dimension], axis=0),jnp.amin(target_desc[:,dimension], axis=0)
            low_limit = jnp.min(curr_min, target_min)
            high_limit = jnp.max(curr_max, target_max)
            histo_target, edges = jnp.histogram(target_desc[:,dimension], bins=100, range=(low_limit,high_limit), density=True)
            histo_curr, edges = jnp.histogram(current_desc[:,dimension], bins=100, range=(low_limit,high_limit), density=True)
            earthmover += wasserstein_distance(edges[:-1], edges[:-1], histo_target, histo_curr)
        return earthmover


