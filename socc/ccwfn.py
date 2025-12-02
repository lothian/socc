"""
ccwfn.py: CC T-amplitude Solver
"""

#if __name__ == "__main__":
#    raise Exception("This file cannot be invoked on its own.")

import psi4
import time
import numpy as np
from opt_einsum import contract
from .hamiltonian import hamiltonian
from .utils import helper_diis, print_wfn, permute_triples
from .cctriples import t_viking_ijk, t_viking_abc, t_viking_ab, t3_ijk, t3_ab
import sys

np.set_printoptions(precision=10, linewidth=300, threshold=sys.maxsize, suppress=True)

class ccwfn(object):
    """
    A spin-orbital CC wave function and energy object.

    Attributes
    ----------
    ref : Psi4 SCF Wavefunction object
        the reference wave function built by Psi4 energy() method
    eref : float
        the energy of the reference wave function (including nuclear repulsion contribution)
    nfzc : int
        the number of frozen core orbitals
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nmo : int
        the number of active orbitals
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix, the ERIs, and various property integrals
    o : NumPy slice
        occupied orbital subspace
    v : NumPy slice
        virtual orbital subspace
    Dia : NumPy array
        one-electron energy denominator
    Dijab : NumPy array
        two-electron energy denominator
    t1 : NumPy array
        T1 amplitudes
    t2 : NumPy array
        T2 amplitudes
    ecc | float
        the final CC correlation energy

    Methods
    -------
    solve_cc()
        Solves the CC T amplitude equations
    residuals()
        Computes the T1 and T2 residuals for a given set of amplitudes and Fock operator
    """

    def __init__(self, scf_wfn, **kwargs):
        """
        Parameters
        ----------
        scf_wfn : Psi4 Wavefunction Object
            computed by Psi4 energy() method

        Returns
        -------
        None
        """

        time_init = time.time()

        valid_cc_models = ['CCSD', 'CCSD(T)', 'CC3']
        model = kwargs.pop('model','CCSD').upper()
        if model not in valid_cc_models:
            raise Exception("%s is not an allowed CC model." % (model))
        self.model = model

        # Cartesian indices
        cart = {"X":0, "Y":1, "Z":2}

        self.ref = scf_wfn
        self.eref = self.ref.energy()
        self.nfzc = self.ref.nfrzc()               # frozen core MOs (not spin-orbitals)
        nao = self.nao = self.ref.nalpha() - self.nfzc        # active alpha occupied spin-orbitals
        nbo = self.nbo = self.ref.nbeta() - self.nfzc         # active beta occupied spin-orbitals
        no = self.no = self.nao + self.nbo                   # active occ spin-orbitals
        nmo = self.nmo = 2*self.ref.nmo()                # all spin-orbitals
        nav = self.nav = self.ref.nmo() - self.nao - self.nfzc     # active alpha virtual spin-orbitals
        nbv = self.nbv = self.ref.nmo() - self.nbo - self.nfzc     # active beta virtual spin-orbitals
        nv = self.nv = self.nav + self.nbv                   # active virt
        nact = self.nact = self.no + self.nv                   # all active MOs

        print("NMO = %d; NACT = %d; NO = %d; NV = %d" % (self.nmo, self.nact, self.no, self.nv))

        # orbital subspaces
        ao = self.ao = slice(0,nao)
        bo = self.bo = slice(nao,no)
        av = self.av = slice(no,no+nav)
        bv = self.bv = slice(no+nav,nact)
        o = self.o = slice(0,no)
        v = self.v = slice(no,nact)

        # spin-orbital --> spin mapping
        spin = np.zeros((nact), dtype=int)
        spin[bo] = 1
        spin[bv] = 1
        self.spin = spin

        # spin-orbital --> spatial-orbital mapping
        spat = np.zeros((nact), dtype=int)
        spat[ao] = np.arange(nao)
        spat[bo] = np.arange(nbo)
        spat[av] = np.arange(nao,nao+nav)
        spat[bv] = np.arange(nbo,nbo+nbv)

        # Get MOs
        Ca = self.ref.Ca_subset("AO", "ACTIVE")
        Cb = self.ref.Cb_subset("AO", "ACTIVE")

        self.field = kwargs.pop('field', False)
        if self.field is True:
            field_strength = kwargs.pop('field_strength', 0.0)
            field_axis = kwargs.pop('field_axis', 'Z').upper()
            if field_axis not in cart:
                raise Exception("Only X, Y, or Z are allowed field axes.")
            print("Adding %8.5f dipole field along %s axis to Hamiltonian." % (field_strength, field_axis))
            field_axis = cart[field_axis]
            self.H = hamiltonian(self.ref, Ca, Cb, spin, spat, field_strength=field_strength, field_axis=field_axis)
        else:
            self.H = hamiltonian(self.ref, Ca, Cb, spin, spat)

        # denominators
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # first-order amplitudes
        self.t1 = np.zeros((self.no, self.nv))
        self.t2 = self.H.ERI[o,o,v,v]/self.Dijab

        # Compute MP2 energy as a test
        emp2 = (1/4) * contract('ijab,ijab->', self.t2, self.H.ERI[o,o,v,v])
        print("E(MP2) = %20.15f" % (emp2))

        print("CCWFN object initialized in %.3f seconds." % (time.time() - time_init))


    def solve_cc(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1, **kwargs):
        """
        Parameters
        ----------
        e_conv : float
            convergence condition for correlation energy (default if 1e-7)
        r_conv : float
            convergence condition for wave function rmsd (default if 1e-7)
        maxiter : int
            maximum allowed number of iterations of the CC equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        ecc : float
            CC correlation energy
        """
        ccsd_tstart = time.time()

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
        F = self.H.F
        ERI = self.H.ERI
        Dia = self.Dia
        Dijab = self.Dijab

        valid_t_algorithms = ['IJK', 'ABC', 'AB']
        self.t_alg = kwargs.pop('alg','IJK').upper()
        if self.t_alg not in valid_t_algorithms:
            raise Exception("%s is not an allowed (T) algorithm." % (self.t_alg))

        self.store_triples = kwargs.pop('store_triples', False)
        if self.store_triples is True:
            print("Triples tensors will be stored in full.")
            self.t3 = np.zeros((no, no, no, nv, nv, nv))
        elif self.field is True and self.model == 'CC3':
            raise Exception("External fields require full storage of triples in CC3 energy calculations.")

        ecc = self.cc_energy(o, v, F, ERI, self.t1, self.t2)
        print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(self.t1, self.t2, max_diis)

        for niter in range(1, maxiter+1):

            ecc_last = ecc

            r1, r2 = self.residuals(self.t1, self.t2)

            self.t1 += r1/Dia
            self.t2 += r2/Dijab
            rms = contract('ia,ia->', r1/Dia, r1/Dia)
            rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
            rms = np.sqrt(rms)

            ecc = self.cc_energy(o, v, F, ERI, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and abs(rms) < r_conv):
                print("\nCCWFN converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                if (self.model == 'CCSD(T)'):
                    print("E(CCSD) = %20.15f" % ecc)
                    if t_alg == 'IJK':
                        print("Using IJK-driven algorithm.")
                        et = t_viking_ijk(o, v, self.t1, self.t2, F, ERI)
                    elif t_alg == 'ABC':
                        print("Using ABC-driven algorithm.")
                        et = t_viking_abc(o, v, self.t1, self.t2, F, ERI)
                    else:
                        print("Using AB-driven algorithm.")
                        et = t_viking_ab(o, v, self.t1, self.t2, F, ERI)
                    print("E(T)    = %20.15f" % et)
                    ecc = ecc + et
                else:
                    print("E(%s) = %20.15f" % (self.model, ecc))
                self.ecc = ecc
                print("E(TOT)  = %20.15f" % (ecc + self.eref))
                print("\nLargest T Amplitudes:")
                print_wfn(self.t1, self.t2)

                return ecc

            diis.add_error_vector(self.t1, self.t2)
            if niter >= start_diis:
                self.t1, self.t2 = diis.extrapolate(self.t1, self.t2)

    def residuals(self, t1, t2):
        """
        Parameters
        ----------
        t1: NumPy array
            Current T1 amplitudes
        t2: NumPy array
            Current T2 amplitudes

        Returns
        -------
        r1, r2: NumPy arrays
            New T1 and T2 residuals: r_mu = <mu|HBAR|0>
        """

        o = self.o
        v = self.v
        no = self.no
        nv = self.nv
        F = self.H.F
        ERI = self.H.ERI

        Fae = self.build_Fae(o, v, F, ERI, t1, t2)
        Fmi = self.build_Fmi(o, v, F, ERI, t1, t2)
        Fme = self.build_Fme(o, v, F, ERI, t1)
        Wmnij = self.build_Wmnij(o, v, ERI, t1, t2)
        Wabef = self.build_Wabef(o, v, ERI, t1, t2)
        Wmbej = self.build_Wmbej(o, v, ERI, t1, t2)

        r1 = self.r_T1(o, v, F, ERI, t1, t2, Fae, Fme, Fmi)
        r2 = self.r_T2(o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wabef, Wmbej)

        if self.model == 'CC3':
            if self.store_triples is True:
                if self.field is True:
                    x1, x2 = self.CC3_full(o, v, self.H.F0, ERI, Fme, t1, t2, self.H.V)
                else:
                    x1, x2 = self.CC3_full(o, v, F, ERI, Fme, t1, t2)
            else:
                x1, x2 = self.CC3(o, v, F, ERI, Fme, t1, t2, self.t_alg)

            r1 += x1; r2 += x2

        return r1, r2

    def build_taut(self, t1, t2):
        return t2 + (1/2) * (contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1))


    def build_tau(self, t1, t2):
        return t2 + (contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1))


    def build_Fae(self, o, v, F, ERI, t1, t2):
        Fae = F[v,v].copy()
        Fae -= (1/2)*contract('me,ma->ae', F[o,v], t1)
        Fae += contract('mf,mafe->ae', t1, ERI[o,v,v,v])
        Fae -= (1/2)*contract('mnaf,mnef->ae', self.build_taut(t1, t2), ERI[o,o,v,v])
        return Fae


    def build_Fmi(self, o, v, F, ERI, t1, t2):
        Fmi = F[o,o].copy()
        Fmi += (1/2)*contract('me,ie->mi', F[o,v], t1)
        Fmi += contract('ne,mnie->mi', t1, ERI[o,o,o,v])
        Fmi += (1/2)*contract('inef,mnef->mi', self.build_taut(t1, t2), ERI[o,o,v,v])
        return Fmi


    def build_Fme(self, o, v, F, ERI, t1):
        Fme = F[o,v].copy()
        Fme += contract('nf,mnef->me', t1, ERI[o,o,v,v])
        return Fme


    def build_Wmnij(self, o, v, ERI, t1, t2):
        Wmnij = ERI[o,o,o,o].copy()
        Wmnij += contract('je,mnie->mnij', t1, ERI[o,o,o,v]) - contract('ie,mnje->mnij', t1, ERI[o,o,o,v])
        Wmnij += (1/4) * contract('ijef,mnef->mnij', self.build_tau(t1, t2), ERI[o,o,v,v])
        return Wmnij


    def build_Wabef(self, o, v, ERI, t1, t2):
        Wabef = ERI[v,v,v,v].copy()
        Wabef -= contract('mb,amef->abef', t1, ERI[v,o,v,v]) - contract('ma,bmef->abef', t1, ERI[v,o,v,v])
        Wabef += (1/4)*contract('mnab,mnef->abef', self.build_tau(t1, t2), ERI[o,o,v,v])
        return Wabef


    def build_Wmbej(self, o, v, ERI, t1, t2):
        Wmbej = ERI[o,v,v,o].copy()
        Wmbej += contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
        Wmbej -= contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
        Z = (1/2) * t2 + contract('jf,nb->jnfb', t1, t1)
        Wmbej -= contract('jnfb,mnef->mbej', Z, ERI[o,o,v,v])
        return Wmbej


    def r_T1(self, o, v, F, ERI, t1, t2, Fae, Fme, Fmi):
        r1 = F[o,v].copy()
        r1 += contract('ie,ae->ia', t1, Fae)
        r1 -= contract('ma,mi->ia', t1, Fmi)
        r1 += contract('imae,me->ia', t2, Fme)
        r1 -= contract('nf,naif->ia', t1, ERI[o,v,o,v])
        r1 -= (1/2)*contract('imef,maef->ia', t2, ERI[o,v,v,v])
        r1 -= (1/2)*contract('mnae,nmei->ia', t2, ERI[o,o,v,o])
        return r1


    def r_T2(self, o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wabef, Wmbej):
        r2 = ERI[o,o,v,v].copy()
        Z = Fae.copy() - (1/2)*contract('mb,me->be', t1, Fme)
        r2 += contract('ijae,be->ijab', t2, Z) - contract('ijbe,ae->ijab', t2, Z)
        Z = Fmi.copy() + (1/2)*contract('je,me->mj', t1, Fme)
        r2 -= contract('imab,mj->ijab', t2, Z) - contract('jmab,mi->ijab', t2, Z)
        r2 += (1/2)*contract('mnab,mnij->ijab', self.build_tau(t1, t2), Wmnij)
        r2 += (1/2)*contract('ijef,abef->ijab', self.build_tau(t1, t2), Wabef)
        r2 += contract('imae,mbej->ijab', t2, Wmbej) - contract('ie,ma,mbej->ijab', t1, t1, ERI[o,v,v,o])
        r2 -= contract('imbe,maej->ijab', t2, Wmbej) - contract('ie,mb,maej->ijab', t1, t1, ERI[o,v,v,o])
        r2 -= contract('jmae,mbei->ijab', t2, Wmbej) - contract('je,ma,mbei->ijab', t1, t1, ERI[o,v,v,o])
        r2 += contract('jmbe,maei->ijab', t2, Wmbej) - contract('je,mb,maei->ijab', t1, t1, ERI[o,v,v,o])
        r2 += contract('ie,abej->ijab', t1, ERI[v,v,v,o]) - contract('je,abei->ijab', t1, ERI[v,v,v,o])
        r2 -= contract('ma,mbij->ijab', t1, ERI[o,v,o,o]) - contract('mb,maij->ijab', t1, ERI[o,v,o,o])
        return r2


    def cc_energy(self, o, v, F, ERI, t1, t2):
        ecc = contract('ia,ia->', F[o,v], t1)
        ecc += (1/4) * contract('ijab,ijab->', t2, ERI[o,o,v,v])
        ecc += (1/2) * contract('ia,jb,ijab->', t1, t1, ERI[o,o,v,v])
        return ecc


    def build_Woooo_CC3(self, o, v, ERI, t1):
        Woooo = ERI[o,o,o,o].copy()
        Woooo += contract('je,mnie->mnij', t1, ERI[o,o,o,v]) - contract('ie,mnje->mnij', t1, ERI[o,o,o,v])
        tau = contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        Woooo += (1/2) * contract('ijef,mnef->mnij', tau, ERI[o,o,v,v])
        return Woooo


    def build_Wovoo_CC3(self, o, v, ERI, t1, Woooo):
        Wovoo = ERI[o,v,o,o].copy()
        Wovoo -= contract('nb,mnij->mbij', t1, Woooo)
        tau = contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        Wovoo += (1/2) * contract('ijef,mbef->mbij', tau, ERI[o,v,v,v])
        Wovoo += contract('ie,mbej->mbij', t1, ERI[o,v,v,o]) - contract('je,mbei->mbij', t1, ERI[o,v,v,o])
        return Wovoo


    def build_Wooov_CC3(self, o, v, ERI, t1):
        Wooov = ERI[o,o,o,v].copy()
        Wooov -= contract('if,mnef->mnie', t1, ERI[o,o,v,v])
        return Wooov


    def build_Wvovv_CC3(self, o, v, ERI, t1):
        Wvovv = ERI[v,o,v,v].copy()
        Wvovv -= contract('na,nmef->amef', t1, ERI[o,o,v,v])
        return Wvovv


    def build_Wvvvo_CC3(self, o, v, ERI, t1):
        Z1 = contract('if,amef->amei', t1, ERI[v,o,v,v])
        Z2 = ERI[o,o,v,o].copy() + contract('if,mnef->mnei', t1, ERI[o,o,v,v])
        tau = contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        Wvvvo = ERI[v,v,v,o].copy()
        Wvvvo += contract('if,abef->abei', t1, ERI[v,v,v,v])
        Wvvvo -= contract('mb,amei->abei', t1, Z1) - contract('ma,bmei->abei', t1, Z1)
        Wvvvo += (1/2) * contract('mnei,mnab->abei', Z2, tau)
        Wvvvo -= contract('ma,mbei->abei', t1, ERI[o,v,v,o]) - contract('mb,maei->abei', t1, ERI[o,v,v,o])
        return Wvvvo


    def CC3(self, o, v, F, ERI, Fme, t1, t2, alg='IJK'):
        Woooo = self.build_Woooo_CC3(o, v, ERI, t1)
        Wovoo = self.build_Wovoo_CC3(o, v, ERI, t1, Woooo)
        Wooov = self.build_Wooov_CC3(o, v, ERI, t1)
        Wvovv = self.build_Wvovv_CC3(o, v, ERI, t1)
        Wvvvo = self.build_Wvvvo_CC3(o, v, ERI, t1)

        if alg == 'IJK':
            x1 = np.zeros_like(t1)
            x2 = np.zeros_like(t2)
            no = x1.shape[0]
            for i in range(no):
                for j in range(no):
                    for k in range(no):
                        t3 = t3_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)

                        x1[i] += (1/4)*contract('bc,abc->a', ERI[j,k,v,v], t3)
                        x2[i,j] += contract('c,abc->ab', Fme[k], t3)
                        x2[i,j] += (1/2)*contract('dbc,abc->ad', Wvovv[:,k,:,:], t3)
                        x2[i,j] -= (1/2)*contract('abc,dbc->ad', Wvovv[:,k,:,:], t3)
                        for l in range(no):
                            tmp = (1/2)*contract('c,abc->ab', Wooov[j,k,l,:], t3)
                            x2[i,l] -= tmp
                            x2[l,i] += tmp

            return x1, x2

        elif alg == 'AB':
            x1 = np.zeros_like(t1.T)
            x2 = np.zeros_like(t2.T)
            nv = x1.shape[0]
            no = x1.shape[1]
            for a in range(nv):
                for b in range(nv):
                    ijkc = t3_ab(o, v, a, b, t2, F, Wvvvo, Wovoo)

                    x1[a] += (1/4) * contract('ijkc,jkc->i', ijkc, ERI[o,o,b+no,v])
                    x2[a,b] += contract('kc,ijkc->ij', Fme, ijkc)
                    tmp = -(1/2) * contract('ijkc,jklc->il', ijkc, ERI[o,o,o,v])
                    x2[a,b] += tmp - tmp.swapaxes(0,1)
                    for d in range(nv):
                        tmp = (1/2) * contract('ijkc,kc->ij', ijkc, ERI[d+no,o,b+no,v])
                        x2[a,d] += tmp
                        x2[d,a] -= tmp

            return x1.T, x2.T


    def CC3_full(self, o, v, F, ERI, Fme, t1, t2, V=0):
        Woooo = self.build_Woooo_CC3(o, v, ERI, t1)
        Wovoo = self.build_Wovoo_CC3(o, v, ERI, t1, Woooo)
        Wooov = self.build_Wooov_CC3(o, v, ERI, t1)
        Wvovv = self.build_Wvovv_CC3(o, v, ERI, t1)
        Wvvvo = self.build_Wvvvo_CC3(o, v, ERI, t1)

        # <mu3|[H^,T2]|0>
        tmp = contract('ijad,bcdk->ijkabc', t2, Wvvvo)
        t3 = permute_triples(tmp, 'k/ij', 'a/bc')
        tmp = -contract('ilab,lcjk->ijkabc', t2, Wovoo)
        t3 += permute_triples(tmp, 'i/jk', 'c/ab')

        if self.field is True:
            Voo = V[o,o].copy() + contract('ie,me->mi', t1, V[o,v])
            Vvv = V[v,v].copy() - contract('ma,me->ae', t1, V[o,v])
            # <mu3|[V,T3]|0>
            tmp = contract('ijkabc,dc->ijkabd', self.t3, Vvv)
            t3 += tmp - tmp.swapaxes(3,5) - tmp.swapaxes(4,5)
            tmp = -contract('ijkabc,kl->ijlabc', self.t3, Voo)
            t3 += tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
            # 1/2 <mu3|[[V,T2],T2]|0>
            tmp = contract('lkbc,ld->bcdk', t2, V[o,v])
            tmp = -contract('bcdk,ijad->ijkabc', tmp, t2)
            t3 += permute_triples(tmp, 'k/ij', 'a/bc')

        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        denom = occ.reshape(-1,1,1,1,1,1) + occ.reshape(-1,1,1,1,1) + occ.reshape(-1,1,1,1) - vir.reshape(-1,1,1) - vir.reshape(-1,1) - vir
        t3 = t3/denom

        self.t3 = t3

        x1 = np.zeros_like(t1)
        x2 = np.zeros_like(t2)

        # <mu1|[H,T3]|0>
        x1 = (1/4) * contract('ijkabc,jkbc->ia', self.t3, ERI[o,o,v,v])
        # <mu2|[H,T3]|0>
        x2 = contract('ijkabc,kc->ijab', self.t3, Fme)
        tmp = (1/2) * contract('ijkabc,dkbc->ijad', self.t3, Wvovv)
        x2 += tmp - tmp.swapaxes(2,3)
        tmp = -(1/2) * contract('ijkabc,jklc->ilab', self.t3, Wooov)
        x2 += tmp - tmp.swapaxes(0,1)

        return x1, x2
