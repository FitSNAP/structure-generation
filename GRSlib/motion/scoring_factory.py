#from GRSlib.parallel_tools import ParallelTools
from GRSlib.motion.scoring import Scoring
from GRSlib.motion.lossfunc.moments import Moments
from GRSlib.motion.lossfunc.entropy import Entropy
from GRSlib.motion.lossfunc.wasserstein import Wasserstein
#   from GRSlib.motion.lossfunc.moments import *

# Need to direct the scoring class to the appropiate loss function generator, this will connect
# with what the user defines. Will allow for developers/users to define their own loss function
# the same way Moments and Entropy (soon) are implemented.

def scoring(lossff_name, pt, config):
    """Scoring Factory for custom loss functions"""
    instance = search(lossff_name)
    if config.args.verbose:
        pt.single_print("Calling {} for structure scoring".format(lossff_name))
    
#    attributes = [attr for attr in dir(instance) if not attr.startswith('__')]
#    print("attr of scoring instance:",instance)
#    print(attributes)

    #instance.__init__(pt, config)
    return instance


def search(lossff_name):
    instance = None
    for cls in Scoring.__subclasses__():
        if cls.__name__.lower() == lossff_name.lower():
            instance = Scoring.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in scoring types".format(lossff_name))
    else:
        return instance
