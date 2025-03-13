#
import sys
from opt_tools import *

#-----------------------------------------------------------------------
#initialize ACE fingerprints
from lammps_ACE_basis import *
# define the ACE descriptor set
elements = ['Ta']
# number of bonds in fingerprints (descriptor rank)
ranks = [1,2,3]
# angular character of fingerprints
lmax = [0,3,2]
# minimum l per rank (for trimming descriptor list)
lmin = [0,0,0]
# radial character of fingerprints
nmax = [8,3,3]
#make_ACE_functions(elements,ranks,lmax,lmin,nmax)
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# set target structure (From file)
from opt_tools import *
#av,var = build_target(start)
av = np.load('A_avg.npy')
import sys
from opt_tools import *

#-----------------------------------------------------------------------
#initialize ACE fingerprints
from lammps_ACE_basis import *
# define the ACE descriptor set
elements = ['Ta']
# number of bonds in fingerprints (descriptor rank)
ranks = [1,2,3]
# angular character of fingerprints
lmax = [0,3,2]
# minimum l per rank (for trimming descriptor list)
lmin = [0,0,0]
# radial character of fingerprints
nmax = [8,3,3]
#make_ACE_functions(elements,ranks,lmax,lmin,nmax)
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# set target structure (From file)
from opt_tools import *
#av,var = build_target(start)
av = np.load('A_avg.npy')
var = np.load('A_var.npy')
#-----------------------------------------------------------------------

# GSQS local optimization
#-----------------------------------------------------------------------
import jax.numpy as np
import scipy as sp
import lammps.mliap
from lammps.mliap.loader import *
import pandas as pd
import copy
import pickle
import os
import shutil
import random

#count descriptors from yace file
n_descs=get_desc_count('coupling_coefficients.yace')
elems=get_desc_count('coupling_coefficients.yace',return_elems=True)
nelements = len(elems)

data_path="./StructureDump"

soft_strength=1.0
ml_strength=1.0

# max and minimum sizes for cells
maxcellsize=40
mincellsize=4
target=vnp.zeros((len(mask)))
#composition in dictionary form: e.g. ( W:0.5, Be:0.5 )
target_comps = {'Ta':1.0}
# extra objective function variables to target exact matches of descriptors
q1_wt=5.0
q1_cross_wt=5.0
q2_wt=0.0
q2_cross_wt=0.0
# flag to perform minimization with variable cell lengths
box_rlx = False
# flag to run at temperature first
run_temp = False
fire_min = False
# flag to use linestyle quadratic minimization
line_min = True
randomize_comps=False
if not fire_min and box_rlx:
    min_typ_global = 'box'
else:
    min_typ_global = 'min'

#for now, read in target distribution (average and variance)
descriptors_target_1 = np.load('A_avg.npy')#'target_descriptors.npy')
descriptors_target_2 = np.load('A_var.npy')#'target_var_descriptors.npy')

class GSQSModel:
    def __init__(self, n_elements, n_descriptors_tot, mask):
        self.mask=mask
        self.n_descriptors=n_descriptors_tot
        self.n_descriptors_keep=len(mask)*n_elements
        self.n_elements=n_elements
        self.n_params=1
        self.sum=vnp.zeros((self.n_descriptors_keep,))
        self.sumsq=vnp.zeros((self.n_descriptors_keep,))
        self.q_count=0
        self.first_moment_grad=grad(self.first_moment)
        self.V_grad=grad(self.V)
        self.K_self=self_weight
        self.K_cross=cross_weight
        self.first_mom_weight_cross = q1_cross_wt
        self.first_mom_weight = q1_wt
        self.second_mom_weight_cross = q2_cross_wt
        self.second_mom_weight = q2_wt
        self.whitening=np.identity(self.n_descriptors_keep)
        self.mode=  "update"
        self.data=[]

    def set_mode_update(self):
        self.mode="update"

    def set_mode_run(self):
        self.mode="run"

    def update(self,d):
        if self.n_elements > 1:
            dft=d.flatten()
        else:
            dft = d
        self.q_count += dft.shape[0]
        self.sum+=vnp.sum(dft,axis=0)
        self.sumsq+=vnp.sum(dft*dft,axis=0)
        self.set_mode_run()

    #match for mean descriptor value for the set of structures
    @partial(jit, static_argnums=(0))
    def first_moment_cross(self, descriptors):
        #n_atoms=descriptors.shape[0]
        #n_descriptors=descriptors.shape[1]
        avgs = self.sum/self.q_count
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        abs_diffs_1 = np.nan_to_num(abs_diffs_1)
        is_zero = np.isclose(abs_diffs_1,np.zeros(abs_diffs_1.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight_cross)
        tst_residual_1 = np.sum(abs_diffs_1) +bonus
        return tst_residual_1

    #match for variance of descriptor value for the set of structures
    @partial(jit, static_argnums=(0))
    def second_moment_cross(self, descriptors):
        #n_atoms=descriptors.shape[0]
        #n_descriptors=descriptors.shape[1]
        vrs = self.sumsq/self.q_count
        abs_diffs_2 = np.abs(vrs - descriptors_target_2)
        abs_diffs_2 = np.array(abs_diffs_2)
        abs_diffs_2 = np.nan_to_num(abs_diffs_2)
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight_cross)
        tst_residual_2 = np.sum(abs_diffs_2) +bonus
        #print (tst_residual_2)
        return tst_residual_2

    # match of mean descriptor values within the current structure only
    @partial(jit, static_argnums=(0))
    def first_moment(self, descriptors):
        #n_atoms=descriptors.shape[0]
        #n_descriptors=descriptors.shape[1]
        avgs = np.average(descriptors,axis=0)
        abs_diffs_1 = np.abs(avgs - descriptors_target_1)
        abs_diffs_1 = np.array(abs_diffs_1)
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.first_mom_weight)
        tst_residual_1 = np.sum(abs_diffs_1) +bonus

        #n_descriptors=descriptors.shape[1]
        is_zero = np.isclose(abs_diffs_2,np.zeros(abs_diffs_2.shape))
        is_zero = np.array(is_zero,dtype=int)
        bonus=-np.sum(is_zero*self.second_mom_weight)
        tst_residual_2 = np.sum(abs_diffs_2) +bonus
        #print (tst_residual_2)
        return tst_residual_2

    @partial(jit, static_argnums=(0))
    def V(self,descriptors,weights=[1.0,0.0]):
        if self.n_elements >= 1:
            descriptors_flt = descriptors.flatten()
        else:
            descriptors_flt = descriptors
        return self.K_self*vi + self.K_cross*vj

    def __call__(self, elems, descriptors, beta, energy):
        self.last_descriptors=descriptors.copy()
        if self.mode=="run":
            b=descriptors[:,self.mask]
            ener=self.V(b)
            energy[:]=0
            energy[0]=ener
            b=self.V_grad(b)
            if not np.all(np.isfinite(b)):
                print("GRAD ERROR!")
            beta[:,:]=0
            beta[:,self.mask]=b

            self.update(b)

class GSQSSampler:
    def __init__(self, model, before_loading_init):
        self.model=model
        self.lmp = lammps.lammps(cmdargs=['-screen','none'])
        lammps.mliap.activate_mliappy(self.lmp)
        self.lmp.commands_string(before_loading_init)
        lammps.mliap.load_model(em)

    def update_model(self):
        self.model.set_mode_update()
        self.lmp.commands_string("variable etot equal etotal")
        self.lmp.commands_string("variable ptot equal press")
        self.lmp.commands_string("variable pairp equal epair")
        self.lmp.commands_string("variable numat equal atoms")
        self.model.set_mode_run()

        else:
            self.lmp.commands_string(cmd)

#begin the actual optimization loop
    pass
try:
lmpi = lammps.lammps(cmdargs=['-screen','none'])
activate_mliappy(lmpi)
sampler.update_model()

    if not randomize_comps:
    else:
        target_comps_rnd = rand_comp(target_comps)
    sampler=GSQSSampler(em,g)
    em.K_cross=cross_weight
    em.K_self=self_weight
    if run_temp:
        sampler.run("run 1000")
    if fire_min:
        sampler.run("min_style  fire")
        sampler.run("""min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no""")
    if line_min:
        sampler.run("min_style  cg")
        sampler.run("min_modify  dmax 0.05 line quadratic")


    #sampler.run("minimize 1e-12 1e-12 10000 100000")
    #sampler.run("minimize 1e-6 1e-6 10000 100000")
    #sampler.run("minimize 1e-6 1e-6 1000 10000")
    sampler.run("minimize 1e-3 1e-3 100 10000")
    sampler.run("write_data %s/sample.%i.dat " % (data_path,i) )
    sampler.update_model()

    i+=1


