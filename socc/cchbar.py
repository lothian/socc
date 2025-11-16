"""
cchbar.py: Builds the similarity-transformed Hamiltonian (one- and two-body terms only).
"""

#if __name__ == "__main__":
#    raise Exception("This file cannot be invoked on its own.")


import time
import numpy as np
from opt_einsum import contract


class cchbar(object):
    """
    An spin-orbital CC Similarity-Transformed Hamiltonian object.

    Attributes
    ----------
    Hov : NumPy array
        The occupied-virtual block of the one-body component HBAR.
    Hvv : NumPy array
        The virtual-virtual block of the one-body component HBAR.
    Hoo : NumPy array
        The occupied-occupied block of the one-body component HBAR.
    Hoooo : NumPy array
        The occ,occ,occ,occ block of the two-body component HBAR.
    Hvvvv : NumPy array
        The vir,vir,vir,vir block of the two-body component HBAR.
    Hvovv : NumPy array
        The vir,occ,vir,vir block of the two-body component HBAR.
    Hooov : NumPy array
        The occ,occ,occ,vir block of the two-body component HBAR.
    Hovvo : NumPy array
        The occ,vir,vir,occ block of the two-body component HBAR.
    Hovov : NumPy array
        The occ,vir,occ,vir block of the two-body component HBAR.
    Hvvvo : NumPy array
        The vir,vir,vir,occ block of the two-body component HBAR.
    Hovoo : NumPy array
        The occ,vir,occ,occ block of the two-body component HBAR.

    """
    def __init__(self, ccwfn):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            amplitudes instantiated to defaults or converged

        Returns
        -------
        None
        """

        time_init = time.time()

        self.ccwfn = ccwfn

        F = ccwfn.H.F
        ERI = ccwfn.H.ERI
        t1 = ccwfn.t1
        t2 = ccwfn.t2
        o = self.o = ccwfn.o
        v = self.v = ccwfn.v
        self.no = ccwfn.no
        self.nv = ccwfn.nv

        self.Hov = self.build_Hov(o, v, F, ERI, t1)
        self.Hvv = self.build_Hvv(o, v, F, ERI, self.Hov, t1, t2)
        self.Hoo = self.build_Hoo(o, v, F, ERI, self.Hov, t1, t2)
        self.Hoooo = self.build_Hoooo(o, v, ERI, t1, t2)
        self.Hvvvv = self.build_Hvvvv(o, v, ERI, t1, t2)
        self.Hvovv = self.build_Hvovv(o, v, ERI, t1)
        self.Hooov = self.build_Hooov(o, v, ERI, t1)
        self.Hovvo = self.build_Hovvo(o, v, ERI, t1, t2)
        Zovov = self.build_Zovov(o, v, ERI, t2)
        self.Hvvvo = self.build_Hvvvo(o, v, ERI, self.Hov, self.Hvvvv, Zovov, t1, t2)
        self.Hovoo = self.build_Hovoo(o, v, ERI, self.Hov, self.Hoooo, Zovov, t1, t2)

        print("\nHBAR constructed in %.3f seconds." % (time.time() - time_init))

    def build_Hov(self, o, v, F, ERI, t1):
        Hov = F[o,v].copy()
        Hov += contract('nf,mnef->me', t1, ERI[o,o,v,v])
        return Hov


    def build_Hvv(self, o, v, F, ERI, Hov, t1, t2):
        Hvv = F[v,v].copy()
        Hvv -= (1/2) * contract('me,ma->ae', F[o,v], t1)
        Hvv -= (1/2) * contract('me,ma->ae', Hov, t1)
        Hvv += contract('mf,amef->ae', t1, ERI[v,o,v,v])
        Hvv -= (1/2) * contract('mnaf,mnef->ae', self.ccwfn.build_taut(t1, t2), ERI[o,o,v,v])
        return Hvv


    def build_Hoo(self, o, v, F, ERI, Hov, t1, t2):
        Hoo = F[o,o].copy()
        Hoo += (1/2) * contract('ie,me->mi', t1, F[o,v])
        Hoo += (1/2) * contract('ie,me->mi', t1, Hov)
        Hoo += contract('ne,mnie->mi', t1, ERI[o,o,o,v])
        Hoo += (1/2) * contract('inef,mnef->mi', self.ccwfn.build_taut(t1, t2), ERI[o,o,v,v])
        return Hoo


    def build_Hoooo(self, o, v, ERI, t1, t2):
        Hoooo = ERI[o,o,o,o].copy()
        Hoooo += contract('je,mnie->mnij', t1, ERI[o,o,o,v]) - contract('ie,mnje->mnij', t1, ERI[o,o,o,v])
        Hoooo += (1/2) * contract('ijef,mnef->mnij', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v]) 
        return Hoooo


    def build_Hvvvv(self, o, v, ERI, t1, t2):
        Hvvvv = ERI[v,v,v,v].copy()
        Hvvvv -= contract('mb,amef->abef', t1, ERI[v,o,v,v]) - contract('ma,bmef->abef', t1, ERI[v,o,v,v])
        Hvvvv += (1/2) * contract('mnab,mnef->abef', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])
        return Hvvvv


    def build_Hvovv(self, o, v, ERI, t1):
        Hvovv = ERI[v,o,v,v].copy()
        Hvovv -= contract('na,nmef->amef', t1, ERI[o,o,v,v])
        return Hvovv


    def build_Hooov(self, o, v, ERI, t1):
        Hooov = ERI[o,o,o,v].copy()
        Hooov += contract('if,mnfe->mnie', t1, ERI[o,o,v,v])
        return Hooov


    def build_Hovvo(self, o, v, ERI, t1, t2):
        Hovvo = ERI[o,v,v,o].copy()
        Hovvo += contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
        Hovvo -= contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
        tau = t2 + contract('ia,jb->ijab', t1, t1)
        Hovvo -= contract('jnfb,mnef->mbej', tau, ERI[o,o,v,v])
        return Hovvo


    def build_Zovov(self, o, v, ERI, t2):
        return ERI[o,v,o,v].copy() + contract('nibf,mnef->mbie', t2, ERI[o,o,v,v])


    def build_Hvvvo(self, o, v, ERI, Hov, Hvvvv, Zovov, t1, t2):
        Hvvvo = ERI[v,v,v,o].copy()
        Hvvvo -= contract('me,miab->abei', Hov, t2)
        Hvvvo += contract('if,abef->abei', t1, Hvvvv)
        Hvvvo += (1/2) * contract('mnab,mnei->abei', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
        Hvvvo -= contract('miaf,mbef->abei', t2, ERI[o,v,v,v]) - contract('mibf,maef->abei', t2, ERI[o,v,v,v])
        Hvvvo += contract('ma,mbie->abei', t1, Zovov) - contract('mb,maie->abei', t1, Zovov)
        return Hvvvo


    def build_Hovoo(self, o, v, ERI, Hov, Hoooo, Zovov, t1, t2):
        Hovoo = ERI[o,v,o,o].copy()
        Hovoo -= contract('me,ijbe->mbij', Hov, t2)
        Hovoo -= contract('nb,mnij->mbij', t1, Hoooo)
        Hovoo += (1/2) * contract('ijef,mbef->mbij', self.ccwfn.build_tau(t1, t2), ERI[o,v,v,v])
        Hovoo += contract('jnbe,mnie->mbij', t2, ERI[o,o,o,v]) - contract('inbe,mnje->mbij', t2, ERI[o,o,o,v])
        Hovoo -= contract('ie,mbje->mbij', t1, Zovov) - contract('je,mbie->mbij', t1, Zovov)
        return Hovoo
