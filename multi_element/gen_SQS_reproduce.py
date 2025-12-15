from mpi4py import MPI
from GRSlib.GRS import GRS
import random, copy, os, glob, shutil
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

settings = \
{
"BASIS":
    {
    "descriptor": "ACE",
    "numTypes": 2,
    "elements": "Ca Mg", 
    "rcutfac": '4.6 4.5 4.5 4.9',
    "lambda": "0.046 0.045 0.045 0.049",
    "rcinner":"0.0 0.0 0.0 0.0",
    "drcinner":"0.0 0.0 0.0 0.0",
    "ranks": "1 2 3 4",
    "lmax": "0 1 2 1",
    "lmin": "0 0 2 1",
    "nmax": "4 2 1 1",
    "nmaxbase": 4,
    "bzeroflag": 0
    },
"SCORING":
    {
    "score_type": "moments",
    "strength_target": 1.0, 
    "strength_prior": 0.0, 
    "norm_by_numdesc": True,
    "moments": "mean" ,
    "moments_coeff": "1.0",
    "moments_bonus": "0 " ,
    },
"TARGET":
    {
    "target_fname": "supercell_target.data",
    "start_fname": "starting.data",
    "job_prefix": "REPSQS"
    },
"GRADIENT":
    {
    "soft_strength": 0.0,
    "ml_strength": 1.0,
    "nsteps": 10000,
    "temperature": 1.0,
    "min_type": "none"
    },
"GENETIC":
    {
    "start_type": "template",  #Can be random or template right now. If template, starting generation is ["TARGET"].start_fname
    "mutation_rate": 1.0,
    "mutation_types": {"perturb": 0.0, "change_ele": 1.0, "atom_count" : 0.0, "volume" : 0.0, "minimize" : 0.0, "ortho_cell" : 0.0}, 
    "population_size": 10,
    "ngenerations": 10,
    "tournament_size": 2,
    "replacements": True,
    "multi_mutate": True,
    #"max_atoms": 50,
    #"min_atoms": 10,
    "density_ratio": 1.0,
    "composition_constraint": {'Ca':0.25, 'Mg':0.75}
    }
}

grs = GRS(settings,comm=comm)

#score = grs.get_score(settings["TARGET"]["start_fname"])
#print("     Starting Score:",score)

updated_struct = settings["TARGET"]["start_fname"]

grs.set_prior([updated_struct])

scores, best_struct = grs.genetic_move(updated_struct)

#updated_struct = grs.gradient_move(best_struct)
#score = grs.get_score(updated_struct)
#print("     Ending Score:",score)

#updated_struct = grs.update_start(updated_struct,"MinScore")
#grs.set_prior(glob.glob(settings['TARGET']["job_prefix"]+"*.data"))

exit()
