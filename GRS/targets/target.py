
class Target:
    def __init__(self,t,fingerprints,typ='file'):
        self.target_type = typ
        self.t = t
        self.fingerprints=fingerprints
        self.target = None
    
    def get_target_dist(self):
        if self.fingerprints.fingerprint_type == 'ace':
            self.Apot = self.fingerprints.Apot
            self.labels=Apot.nus
    