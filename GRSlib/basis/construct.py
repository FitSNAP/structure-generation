#placeholder for ACE basis constructor, but this could also be any basis set.
#placeholder
#NOTE you must have FitSNAP in your python path to use this
import pickle
from default_ACE_settings import *
def make_ACE_functions(elements,ranks,lmax,lmin,nmax):
    #        elements = ['Cr','Fe','Si','V']
        # ranks of descriptors
        #ranks = [1,2,3]
        # angular character of fingerprints
        #lmax = [0,2,2]
        # minimum l per rank (for trimming descriptor list)
        #lmin = [0,0,1]
        # radial character of fingerprints
        #nmax = [3,2,1]
    try:
        from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import AcePot
        #flag to use generalized CG coefficients instead of generalized Wigner symbols
        # FitSNAP default is 'False'.
        cg_flag = True

        reference_ens = [0.]*len(elements)
        nradbase=max(nmax)

        # radial function hyperparameters are defined per bond type:
        bonds = [p for p in itertools.product(elements,elements)]
        #print('bonds',bonds)
        rc_range,rc_default,lmb_default,rcin_default = get_default_settings(elements,nshell=2,return_range=True,apply_shift=False)
        rcutfac = [float(k) for k in rc_default.split()[2:]]
        lmbda = [float(k) for k in lmb_default.split()[2:]]
        #you typically don't want an inner cutoff for structure generation
        rcinner = [0.0] * len(bonds)
        drcinner = [0.0]* len(bonds)


        #-------------------------------------------------------------------------------------
        ldict = {rank:lmax[i] for i,rank in enumerate(ranks)}
        L_R = 0
        M_R= 0

        istr = '%d'
        lstr = ''
        rankstr = ''
        for i,val in enumerate(ranks):
            lstr += istr
            rankstr += istr
        lstr = lstr % tuple(lmax)
        rankstr = rankstr % tuple(ranks)
        from fitsnap3lib.lib.sym_ACE.coupling_coeffs import store_generalized
        if cg_flag:
            from fitsnap3lib.lib.sym_ACE.clebsch_couple import get_cg_coupling
            try:
                with open('cg_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                    ccs = pickle.load(handle)
            except FileNotFoundError:
                ccs = get_cg_coupling(ldict,L_R=L_R)
                #store them for later so they don't need to be recalculated
                store_generalized(ccs, coupling_type='cg',L_R=L_R)
        else:
            from fitsnap3lib.lib.sym_ACE.wigner_couple import get_wig_coupling
            try:
                with open('wig_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                    ccs = pickle.load(handle)
            except FileNotFoundError:
                ccs = get_wig_coupling(ldict,L_R=L_R)
                store_generalized(ccs, coupling_type='wig',L_R=L_R)
        Apot = AcePot(elements,reference_ens,ranks,nmax,lmax,nradbase,rcutfac,lmbda,rcinner,drcinner,lmin=lmin, **{'ccs':ccs[M_R]})
        Apot.write_pot('coupling_coefficients')
        #print(Apot.nus)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Need to have fitsnap3 module in your python path to automatically generate ACE bases")
