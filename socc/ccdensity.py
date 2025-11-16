"""
ccdensity.py: Builds the CC density.
"""

import time
import numpy as np
from opt_einsum import contract

class ccdensity(object):
    """
    A spin-orbital CC Density object.

    Attributes
    ----------
    Dov, Dvo, Dvv, Doo : NumPy arrays
        Blocks of the one-body density partitioned by occupied/virtual spaces.
    """
    def __init__(self, ccwfn, cclambda, onlyone=True):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            contains the necessary T-amplitudes (either instantiated to defaults or converged)
        cclambda : PyCC cclambda object
            Contains the necessary Lambda-amplitudes (instantiated to defaults or converged)
        onlyone : Boolean
            only compute the onepdm if True

        Returns
        -------
        None
        """

        time_init = time.time()

        self.ccwfn = ccwfn
        self.cclambda = cclambda

        o = ccwfn.o
        v = ccwfn.v
        no = ccwfn.no
        nv = ccwfn.nv
        F = ccwfn.H.F
        ERI = ccwfn.H.ERI
        t1 = ccwfn.t1
        t2 = ccwfn.t2
        l1 = cclambda.l1
        l2 = cclambda.l2

        opdm = np.zeros((no+nv, no+nv))
        opdm[o,o] = self.build_Doo(t1, t2, l1, l2)
        opdm[v,v] = self.build_Dvv(t1, t2, l1, l2)
        opdm[o,v] = self.build_Dov(t1, t2, l1, l2)
        opdm[v,o] = self.build_Dvo(l1)

        self.opdm = opdm

        print("\nCCDENSITY initialized in %.3f seconds.\n" % (time.time() - time_init))

    def build_Doo(self, t1, t2, l1, l2):
        Doo = -1.0 * contract('ie,je->ij', t1, l1)
        Doo -= (1/2) * contract('imef,jmef->ij', t2, l2)
        return Doo


    def build_Dvv(self, t1, t2, l1, l2):
        Dvv = contract('mb,ma->ab', t1, l1)
        Dvv += (1/2) * contract('mnbe,mnae->ab', t2, l2)
        return Dvv


    def build_Dvo(self, l1):
        return l1.T.copy()


    def build_Dov(self, t1, t2, l1, l2):  # complete
        Dov = t1.copy()
        Dov += contract('me,imae->ia', l1, t2)
        Dov -= contract('me,ie,ma->ia', l1, t1, t1)
        tmp = contract('mnef,mnaf->ea', l2, t2)
        Dov -= (1/2) * contract('ea,ie->ia', tmp, t1)
        tmp = contract('mnef,inef->mi', l2, t2)
        Dov -= (1/2) * contract('mi,ma->ia', tmp, t1)
        return Dov
