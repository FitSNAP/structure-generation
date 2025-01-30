from GRS.lib_settings import *
from GRS.fingerprints.fingerprint import Fingerprints
from GRS.fingerprints.default_ACE_settings import *
import itertools,pickle

class ACE_ChemPos(Fingerprints):
    def __init__(self,typ,fingerprint_settings):
        super().__init__(typ,fingerprint_settings)
        self.labels = []
        self.verbose=False
        self.Apot = None
        self.fingerprint_settings = fingerprint_settings
        self.required_params = ['elements', 'ranks', 'nmax', 'lmax']
        self.optional_params = ['lmin', 'L_R', 'M_R']
        self.all_params = self.required_params + self.optional_params

    def enumerate_labels(self):
        assert all([ keyword in self.all_params for keyword in list(self.fingerprint_settings.keys())]),"need kewords from required or optional lists"
        elements = self.fingerprint_settings['elements']
        ranks = self.fingerprint_settings['ranks']
        nmax = self.fingerprint_settings['nmax']
        lmax = self.fingerprint_settings['lmax']

        try:
            lmin = self.fingerprint_settings['lmin']
        except KeyError:
            lmin = [0]*len(ranks)
        try:
            L_R = self.fingerprint_settings['L_R']
        except KeyError:
            L_R = 0
        try:
            M_R = self.fingerprint_settings['M_R']
        except KeyError:
            M_R = 0
        

        # FitSNAP default is 'False'.
        cg_flag = True

        reference_ens = [0.]*len(elements)
        nradbase=max(nmax)

        # radial function hyperparameters are defined per bond type:
        bonds = [p for p in itertools.product(elements,elements)]

        if self.verbose:
            print('bonds',bonds)
        rc_range,rc_default,lmb_default,rcin_default = get_default_settings(elements,nshell=2,return_range=True,apply_shift=False)
        rcutfac = [float(k) for k in rc_default.split()[2:]]
        lmbda = [float(k) for k in lmb_default.split()[2:]]
        #you typically don't want an inner cutoff for structure generation
        rcinner = [0.0] * len(bonds)
        drcinner = [0.0]* len(bonds)

        #------ using fitsnap:
        try:
            from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import AcePot
            from fitsnap3lib.lib.sym_ACE.clebsch_couple import get_cg_coupling
            from fitsnap3lib.lib.sym_ACE.wigner_couple import get_wig_coupling
            from fitsnap3lib.lib.sym_ACE.coupling_coeffs import store_generalized
        except ModuleNotFoundError:
            raise ModuleNotFoundError("fitsnap3 needs to be in your python path (currently) in order to use ACE fingerprints")

        ldict = {rank:lmax[i] for i,rank in enumerate(ranks)}

        istr = '%d'
        lstr = ''
        rankstr = ''
        for i,val in enumerate(ranks):
            lstr += istr
            rankstr += istr
        lstr = lstr % tuple(lmax)
        rankstr = rankstr % tuple(ranks)
        print('lib_path',lib_path)
        if cg_flag:
            try:
                with open('%s/cg_LR_%d_r%s_lmax%s.pickle' %(lib_path,L_R,rankstr,lstr),'rb') as handle:
                    ccs = pickle.load(handle)
            except FileNotFoundError:
                ccs = get_cg_coupling(ldict,L_R=L_R)
                #store them for later so they don't need to be recalculated
                store_generalized(ccs, coupling_type='cg',L_R=L_R,store_path=lib_path)
        else:
            try:
                with open('%s/wig_LR_%d_r%s_lmax%s.pickle' %(lib_path,L_R,rankstr,lstr),'rb') as handle:
                    ccs = pickle.load(handle)
            except FileNotFoundError:
                ccs = get_wig_coupling(ldict,L_R=L_R)
                store_generalized(ccs, coupling_type='wig',L_R=L_R,store_path=lib_path)
        Apot = AcePot(elements,reference_ens,ranks,nmax,lmax,nradbase,rcutfac,lmbda,rcinner,drcinner,lmin=lmin, **{'ccs':ccs[M_R]})
        self.Apot = Apot
        self.labels=Apot.nus

    def set_labels(self,labels):
        self.labels = labels

    def write_fingerprint_params(self,prefix,fmt='lammps'):
        if fmt == 'lammps':
            try:
                self.Apot.write_pot(prefix)
            except AttributeError:
                raise AttributeError("need to enumerate ACE descriptors before fingerprint params may be saved in LAMMPS format")
        else:
            raise TypeError('exiting b/c format %s was not recognized' % fmt)
            