#from GRSlib.parallel_tools import ParallelTools
#from GRSlib.motion.scoring_factory import scoring
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.genetic import Genetic
from GRSlib.motion.create import *
import numpy as np
import random, shutil, os, glob

# Two types of motion (aka changes) can be applied to a structure 1) Gradients of the loss function (energy/score) 
# yielding continuous changes, or 2) discrete moves that include (atom addition/removal, chemical identities).
# New types can be added as classes if there is need, follow class inheritence of existing motion types.
# These classes require a scoring function, which is inherited after initializing in GRS.py. 

# A key difference between Gradient and Optimize is that the former operates on a single structure at a time, while
# the latter will create a dictionary of structures and their scores to be evaluated. 

class Gradient:

    def __init__(self, pt, config, scoring):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring

    def none_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """run 0
        write_data %s_last.data""" % (self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, data
    
    def fire_min(self,data,tole=0.0,tolf=0.0):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        dump 1 all custom 1 minimize_fire.dump id type x y z fx fy fz
        minimize %1.1E %1.1E %s %s
        write_data %s_last.data""" % (tole,tolf,self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def line_min(self,data,tole=0.0,tolf=1.e-10,dmax=0.5):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        min_style  cg
        min_modify dmax %2.2f line quadratic
        dump 1 all custom 1 minimize_line.dump id type x y z fx fy fz
        displace_atoms all random 0.01 0.01 0.01 12345 units box
        minimize %1.1E %1.1E %s %s
        write_data %s_last.data
        """ % (dmax,tole,tolf, self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def box_min(self,data,tole=0.0,tolf=0.0,dmax=0.5):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom  0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        min_style  cg
        min_modify dmax %2.2f line quadratic
        dump 1 all custom 1 minimize_box.dump id type x y z fx fy fz
        fix box all box/relax aniso 0.0 couple none nreset 50 vmax %2.2f
        minimize %1.1E %1.1E %s %s
        write_data %s_last.data""" % (dmax,dmax,tole,tolf,self.config.sections['GRADIENT'].nsteps, self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

    def temp_min(self,data):
        #Will construct a set of additional commands to send to LAMMPS before scoring
        add_cmds=\
        """delete_atoms overlap 0.3 all all
        compute cluster all cluster/atom 0.3
        compute max all reduce max c_cluster
        variable exit equal c_max
        fix halt all halt 10 v_exit > 1 error soft
        velocity all create %s %s dist gaussian
        fix nve all nve
        fix lan all langevin %s %s 1.0 48279
        run %s
        unfix nve
        unfix lan
        write_data %s_last.data
        min_style  fire
        min_modify integrator eulerexplicit tmax 10.0 tmin 0.0 delaystep 5 dtgrow 1.1 dtshrink 0.5 alpha0 0.1 alphashrink 0.99 vdfmax 100000 halfstepback no initialdelay no
        dump 1 all custom 1 run_minimize.dump id type x y z fx fy fz
        minimize 1e-6 1e-6 %s %s
        write_data %s_last.data""" % (self.config.sections['GRADIENT'].temperature, np.random.randint(low=1, high=99999), 
                                      self.config.sections['GRADIENT'].temperature, self.config.sections['GRADIENT'].temperature, 
                                      self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix, self.config.sections['GRADIENT'].nsteps, 
                                      self.config.sections['GRADIENT'].nsteps, self.config.sections['TARGET'].job_prefix)
        before_score, after_score = self.scoring.add_cmds_before_score(add_cmds,data)
        end_data = self.config.sections['TARGET'].job_prefix + "_last.data"
        return before_score, after_score, end_data

class Optimize:

    def __init__(self, pt, config, scoring, convert):
        self.pt = pt #ParallelTools()
        self.config = config #Config()
        self.scoring = scoring
        self.convert = convert
        self.create = Create(self.pt, self.config, self.convert)
        self.gradmove = Gradient(pt, config, scoring)

    def find_min_max(self,population):
        population.sort(key = lambda x:x[3],reverse=True)
#        mp_inds = {ii:population[ii][1] for ii in range(len(population))}
        return population[0],population[int(len(population))-1]

    def tournament_selection(self,population,iteration,k,replacements,multi_mutate):
        #This version of the tournament will have preset values of k and mutation/crossover decisions
        #which result in more generic pressure to find higher fitness (lower score).
        count_pop = int(len(population))
        scores = np.array([p[3] for p in population])
        tourny_winners = []
        for rounds in range(self.config.sections["GENETIC"].population_size):
            tourny_trials = np.random.choice(range(0,len(population)),k,replace=replacements)
            indicies = list(tourny_trials)
            pool_scores = list(scores[indicies])
            pool_winner = int(np.sort(np.c_[indicies, pool_scores], axis=0)[0][0]) #index of lowest score in this batch
            tourny_winners.append(pool_winner)

        #TODO Need to decide what to do with the tourny winners

        parents = []
        for id in range(int(count_pop/2)):
            parents.append([population[tourny_winners[id]],population[tourny_winners[int(count_pop-id-1)]]])

        #Now setup for the next iteration of the tournament
        population = []
        if np.random.rand() < float(self.config.sections['GENETIC'].mutation_rate):
            for parent_pair in range(len(parents)):
                ase_parent1 = self.convert.lammps_to_ase(parents[parent_pair][0][2]) # index of parent pair, index of parent, name of data-file
                ase_parent2 = self.convert.lammps_to_ase(parents[parent_pair][1][2]) # index of parent pair, index of parent, name of data-file
                child1, score1, child2, score2 = self.genetic.mutation(ase_parent1,ase_parent2,multi_mutate)
                #Could add logic here to compare child scores to parents
                file_name_child1 = self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.lammps-data"%(parent_pair*2+0,iteration)
                file_name_child2 = self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.lammps-data"%(parent_pair*2+1,iteration)

                shutil.move('child1', file_name_child1)
                shutil.move('child2', file_name_child2)
                population.append([iteration, parent_pair*2+0, file_name_child1, score1])
                population.append([iteration, parent_pair*2+1, file_name_child2, score2])
#                print("Parents:",parents[parent_pair][0][3],parents[parent_pair][1][3], "Children:",score1,score2)
        else:
            for parent_pair in range(len(parents)):
                ase_parent1 = self.convert.lammps_to_ase(parents[parent_pair][0][2]) # index of parent pair, index of parent, name of data-file
                ase_parent2 = self.convert.lammps_to_ase(parents[parent_pair][1][2]) # index of parent pair, index of parent, name of data-file
                child1, score1, child2, score2 = self.genetic.crossover_ASE(ase_parent1,ase_parent2) 
                #Could add logic here to compare child scores to parents
                file_name_child1 = self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.lammps-data"%(parent_pair*2+0,iteration)
                file_name_child2 = self.config.sections['TARGET'].job_prefix + "_Cand%sGen%s.lammps-data"%(parent_pair*2+1,iteration)

                shutil.move('child1', file_name_child1)
                shutil.move('child2', file_name_child2)
                population.append([iteration, parent_pair*2+0, file_name_child1, score1])
                population.append([iteration, parent_pair*2+1, file_name_child2, score2])
#                print("Parents:",parents[parent_pair][0][3],parents[parent_pair][1][3], "Children:",score1,score2)

        return population

    def filter_unique(self,selection):
        used = []
        updated = []
        for s in selection:
            if s[1] not in used:
                updated.append(s)
                used.append(s[1])
        return updated     

    def advance_generations(self, data):
        #More of a super function that will call a bunch of the ones below
        #This should be the default since we dont want to send duplicates the crossover/mutation
        self.genetic = Genetic(self.pt, self.config,self.convert,self.scoring,self.gradmove)    
        k = self.config.sections["GENETIC"].tournament_size
        replacements = self.config.sections["GENETIC"].replacements
        multi_mutate =self.config.sections["GENETIC"].multi_mutate
        
        starting_generation = Create.starting_generation(self,data)
        scores = []
        gen_winners = []
        for candidate in range(self.config.sections["GENETIC"].population_size):
            file_name = self.config.sections['TARGET'].job_prefix+"_Cand%sGen%s.lammps-data"%(candidate,'Init')
            lammps_data = self.convert.ase_to_lammps(starting_generation[candidate],file_name)
            #Honestly I would prefer scores as a dictonary of Key:Item pairs, TODO later.
            scores.append(['Init', candidate, file_name, self.scoring.get_score(lammps_data)])
        # We now have the scores from the starting generation which uses a from_(start_type) to populate
        if self.config.sections["GENETIC"].randomseed != None:
            np.random.seed(self.config.sections["GENETIC"].randomseed)

        for iteration in range(self.config.sections['GENETIC'].ngenerations):       
            #Since we have the starting generation, we will move directly to the tourny selection method

            scores = self.tournament_selection(scores,iteration,k,replacements,multi_mutate)
            highest, lowest = self.find_min_max(scores)
            gen_winners.append(scores[lowest[1]])
            print("Iteration:",iteration, "Lowest:",lowest[2],lowest[3], "Highest:",highest[2],highest[3])
            with open("scoring_%s.txt"%self.config.sections['TARGET'].job_prefix, "a") as f:
                print( lowest,highest, file=f)
 
        for file in glob.glob(self.config.sections['TARGET'].job_prefix + "_Cand*Gen*"):          
            if file not in  [row[2] for row in gen_winners]:
                os.remove(file)

        return gen_winners
        #End of tournament returns winners circle list to GRS.py -> (convert.ASEtoLAMMPS + write score output)

    def latin_hyper(self, **kwargs):
        #placeholder for equal sampling accross input space of generated strucutres
        pass

    def sim_anneal(self, **kwargs):
        #placeholder for simulated annealing of generated strucutres
        pass

    def lib_optimizer(self, **kwargs):
        #placeholder, possible for DAKOTA or pyMOO coupling?
        pass
    
