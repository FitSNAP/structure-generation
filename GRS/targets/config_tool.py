import numpy as np
from ase import Atom,Atoms
from ase.io import read,write

class fsnap_atoms():
    def __init__(self):
        topkeys = ['Data', 'PositionsStyle', 'AtomTypeStyle', 'Label', 'StressStyle', 'LatticeStyle', 'EnergyStyle', 'ForcesStyle']
        self.data = None
        self.dataset = {key:None for key in topkeys}
        return None

    #def set_data(self,data):
    #    self.data = data

    def read_json(self,jfile):
        #import json5
        import json
        with open(jfile,'r') as readin:
            #d = json5.load(readin,parse_constant=True)
            d = json.load(readin)
            dsetkeys = d['Dataset'].keys()
            for key in dsetkeys:
                if key != 'Data':
                    self.dataset[key] = d['Dataset'][key]
                elif key == 'Data':
                    self.dataset[key] = d['Dataset'][key]
            self.data = d['Dataset']['Data'][0]

    def set_ASE(self,atoms,magmomflag = False,chargeflag=False,chiflag=False,use_exist=True, **kwargs):
        self.dataset['AtomTypeStyle'] = "chemicalsymbol"
        self.dataset['EnergyStyle'] = "electronvolt"
        self.dataset['ForcesStyle'] = "electronvoltperangstrom"
        self.dataset['Label'] = "ASE generated"
        self.dataset['LatticeStyle'] = "angstrom"
        self.dataset['PositionsStyle'] = "angstrom"
        self.dataset['StressStyle'] = "kB"
        #dkeys = ['Stress', 'Positions', 'Energy', 'AtomTypes', 'Lattice', 'NumAtoms', 'Forces', 'Chis']
        dkeys = ['Stress', 'Positions', 'Energy', 'AtomTypes', 'Lattice', 'NumAtoms', 'Forces']
        extra_dkeys = ['Chis','Charges', 'Coul_Pots', 'VCNSTRS', 'mus','MagneticMoments']
        data = {dkey:None for dkey in dkeys}
        data['Positions'] = atoms.positions.copy().tolist()
        if not use_exist:
            try:
                en = kwargs['Energy']
            except KeyError:
                en = 0.
            try:
                frc = kwargs['Forces']
            except KeyError:
                frc = [[0.0,0.0,0.0]]*len(atoms)
            try:
                stress = kwargs['Stress']
            except KeyError:
                stress= [[0.0,0.0,0.0],
                    [0.0,0.0,0.0],
                    [0.0,0.0,0.0] ]
        else:
            en = self.data['Energy']
            frc = self.data['Forces']
            stress = self.data['Stress']
        data['Stress'] = stress
        data['Energy'] = en
        data['AtomTypes'] = atoms.get_chemical_symbols()
        data['Lattice'] = atoms.get_cell().tolist()
        data['NumAtoms'] = len(atoms)
        data['Forces'] = frc
        for key in extra_dkeys:
            try:
                data[key] = kwargs[key]
            except KeyError:
                pass
        
        self.data = data
        self.dataset['Data'] = [data]


    def read_ase(self,atoms,isfile=False,use_exist=False):
        if isfile:
            atoms = read(atoms)

    #def set_ASE(self,atoms,magmomflag = False,chargeflag=False,chiflag=False,use_exist=True, **kwargs):
        self.dataset['AtomTypeStyle'] = "chemicalsymbol"
        self.dataset['EnergyStyle'] = "electronvolt"
        self.dataset['ForcesStyle'] = "electronvoltperangstrom"
        self.dataset['Label'] = "ASE generated"
        self.dataset['LatticeStyle'] = "angstrom"
        self.dataset['PositionsStyle'] = "angstrom"
        self.dataset['StressStyle'] = "kB"
        dkeys = ['Stress', 'Positions', 'Energy', 'AtomTypes', 'Lattice', 'NumAtoms', 'Forces']
        extra_dkeys = ['Chis','Charges', 'Coul_Pots', 'VCNSTRS', 'mus','MagneticMoments']
        data = {dkey:None for dkey in dkeys}
        data['Positions'] = atoms.positions.copy().tolist()
        if not use_exist:
            try:
                en = atoms.info['dft_energy']/len(atoms)
            except KeyError:
                en = 0.
            try:
                frc = atoms.arrays['dft_force']
            except KeyError:
                frc = [[0.0,0.0,0.0]]*len(atoms)
            try:
                stress = atoms.arrays['dft_virial']
            except KeyError:
                stress= [[0.0,0.0,0.0],
                    [0.0,0.0,0.0],
                    [0.0,0.0,0.0] ]
        else:
            en = self.data['Energy']
            frc = self.data['Forces']
            stress = self.data['Stress']
        data['Stress'] = stress
        data['Energy'] = en
        data['AtomTypes'] = atoms.get_chemical_symbols()
        data['Lattice'] = atoms.get_cell().tolist()
        data['NumAtoms'] = len(atoms)
        data['Forces'] = frc
        for key in extra_dkeys:
            try:
                data[key] = kwargs[key]
            except KeyError:
                pass
        
        self.data = data
        self.dataset['Data'] = [data]

    def get_ASE(self):
        atoms = Atoms(self.data['AtomTypes'])
        atoms.set_positions(self.data['Positions'])
        atoms.set_cell(self.data['Lattice'])
        try:
            atoms.set_initial_charges(self.data['Charges'])
        except KeyError:
            pass
        atoms.set_pbc(True)
        return atoms

    def reset_chems(self,chemmap):
        symbols = self.data['AtomTypes']
        new_symbols = [chemmap[symbol] for symbol in symbols]
        self.data['AtomTypes'] = new_symbols
        self.dataset['Data'] = [self.data]

    def write_JSON(self,name,write_header=True):
        import json
        with open('%s.json'%name,'w') as writeout:
            if write_header:
                writeout.write('# file %s\n' % name)
            json.dump({'Dataset': self.dataset}, writeout,sort_keys=True,indent=2)
