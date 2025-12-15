from GRSlib.io.sections.sections import Section

#pt = ParallelTools()
#output = Output()
class Scoring(Section):

    def __init__(self, name, config, pt,infile, args):
        super().__init__(name, config, pt, infile,args)
        self.allowedkeys = ['score_type','moments', 'moments_coeff', 'moments_bonus', 'strength_target',
                            'strength_prior','exact_distribution','internal_entropy','ensemble_entropy',
                            'norm_by_numdesc','smartmask']
        self._check_section()
        self.score_type = self.get_value("SCORING", "score_type", None)
        self.strength_target = self.get_value("SCORING", "strength_target", 1.0, interpreter="float")
        self.strength_prior = self.get_value("SCORING", "strength_prior", 0.0, interpreter="float")
        self.exact_distribution = self.get_value("SCORING", "exact_distribution", False)
        self.norm_by_numdesc = self.get_value("SCORING", "norm_by_numdesc", False)
        if self.score_type == "moments":
            self.moments = self.get_value("SCORING", "moments", "mean stdev").split()
            #options are : mean stdev skew kurtosis
            self.moments_coeff = self.get_value("SCORING", "moments_coeff", "1.0 0.1").split()
            #Requires one number per moment
            self.moments_bonus = self.get_value("SCORING", "moments_bonus", "0.0 0.0").split()
            #Score reduction if exact moment value is matched
            self.smartmask = self.get_value("SCORING", "smartmask", 0, interpreter="int")
            #Use a subset of descriptors of given count
        elif self.score_type == "wasserstein":
            self.smartmask = self.get_value("SCORING", "smartmask", 0, interpreter="int")
            #Use a subset of descriptors of given count

        elif self.score_type == "entropy":
            self.internal_entropy = self.get_value("SCORING", "internal_entropy", 1.0)
            # Strength of interactions within a single structure. 
            # Positive value prefers uniform atom environments, negative value promotes diversity (higher entropy)
            self.ensemble_entropy = self.get_value("SCORING", "ensemble_entropy", 1.0)
            # Strength of interactions with prior structures.
            # Positive value attracts current structure to priors, negative value promotes diversity (higher entropy)
        self.delete()

