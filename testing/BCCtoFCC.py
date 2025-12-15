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
    "numTypes": 1,
    "elements": "W", #Needs to be a dict?
    "rcutfac": 4.276275664,
    "lambda": 0.7075106628,
    "rcinner": 0.00,
    "drcinner": 0.01,
    "ranks": "1 2 3 4", 
    "lmax": "0 3 3 0",  
    "lmin": "0 0 0 0",  
    "nmax": "8 6 4 2",  
    "nmaxbase": 8,
    "bzeroflag": 0
    },
"SCORING":
    {
    "score_type": "moments",
    "strength_target": 1.0, 
    "strength_prior": 0.0, 
    "moments": "mean stdev" ,
    "moments_coeff": "1.0 0.0",
    "moments_bonus": "0 0" ,
    "norm_by_numdesc": True,
    },
"TARGET":
    {
    "target_fname": "fcc.data",
#    "target_fdesc": "fcc.npy",
    "start_fname": "bcc.data",
    "prior_fdesc": "prior.npy",
    "job_prefix": "BCCtoFCC"
    },
"GRADIENT":
    {
    "soft_strength": 1.0,
    "ml_strength": 1.0,
    "nsteps": 1000,
    "temperature": 100.0,
    "min_type": "box"
    },
"GENETIC":
    {
    "start_type": "phases",  #Can be random or template right now. If template, starting generation is ["TARGET"].start_fname
    "reference_phases": "bcc",
    "frac_pop_per_ref": "1.0",
    "dev_per_ref": "0.2",
    "mutation_rate": 0.5,
    "mutation_types": {"perturb": 0.0, "change_ele": 0.0, "atom_count" : 0.00, "volume" : 1.00, "minimize" : 0.00, "ortho_cell" : 0.00}, 
    "population_size": 100,
    "ngenerations": 2,
    "max_atoms": 50,
    "min_atoms": 10,
    "max_length_aspect": 2.0,
    "max_angle_aspect": 2.0,
    "density_ratio": 1.1,
    "composition_constraint": {'W':1.0}
    }
}

grs = GRS(settings,comm=comm)

tourny_winners = []

score = grs.get_score(settings["TARGET"]["start_fname"])
grs.config.sections['GRADIENT'].min_type = "line"
line_relaxed_struct = grs.gradient_move(settings["TARGET"]["start_fname"])
line_score = grs.get_score(line_relaxed_struct)
grs.config.sections['GRADIENT'].min_type = "temp"
temp_relaxed_struct = grs.gradient_move(settings["TARGET"]["start_fname"])
temp_score = grs.get_score(temp_relaxed_struct)
grs.config.sections['GRADIENT'].min_type = "box"
box_relaxed_struct = grs.gradient_move(settings["TARGET"]["start_fname"])
box_score = grs.get_score(box_relaxed_struct)

a=np.load('BCCtoFCC_last.npy')
print("Desc_Count:",np.shape(a)[1])
print("Scores (Starting, box, line, temp):",score,box_score,line_score,temp_score)

exit()

scores, best_struct = grs.genetic_move(settings["TARGET"]["target_fname"])

score = grs.get_score(best_struct)
print("     Best Score:",score, "from",best_struct)
renamed_best = "Winner-0_"+best_struct
shutil.move(best_struct, renamed_best)
tourny_winners.append(renamed_best)

for file in glob.glob(grs.config.sections['TARGET'].job_prefix + "_Cand*Gen*"):
    if file not in  [row[2] for row in tourny_winners]:
        os.remove(file)

