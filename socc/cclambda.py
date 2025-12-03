"""
cclambda.py: Lambda-amplitude Solver
"""

#if __name__ == "__main__":
#    raise Exception("This file cannot be invoked on its own.")


import numpy as np
import time
from opt_einsum import contract
from .utils import helper_diis, print_wfn, permute_triples
from .cctriples import t3_ijk, l3_ijk, t3_ab, l3_ab


class cclambda(object):
    """
    An spin-orbital CC wave function and energy object.

    Attributes
    ----------
    ccwfn : PyCC ccwfn object
        the coupled cluster T amplitudes and supporting data structures
    hbar : PyCC cchbar object
        the coupled cluster similarity-transformed Hamiltonian
    l1 : NumPy array
        L1 amplitudes
    l2 : NumPy array
        L2 amplitudes

    Methods
    -------
    solve_lambda()
        Solves the CC Lambda amplitude equations
    residuals()
        Computes the L1 and L2 residuals for a given set of amplitudes and Fock operator
    """
    def __init__(self, ccwfn, hbar):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            the coupled cluster T amplitudes and supporting data structures
        hbar : PyCC cchbar object
            the coupled cluster similarity-transformed Hamiltonian

        Returns
        -------
        None
        """

        self.ccwfn = ccwfn
        self.hbar = hbar

        self.l1 = self.ccwfn.t1.copy()
        self.l2 = self.ccwfn.t2.copy()

    def solve_lambda(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=1):
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
        lecc : float
            CC pseudoenergy

        """
        lambda_tstart = time.time()

        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        l1 = self.l1
        l2 = self.l2
        Dia = self.ccwfn.Dia
        Dijab = self.ccwfn.Dijab
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI

        Hov = self.hbar.Hov
        Hvv = self.hbar.Hvv
        Hoo = self.hbar.Hoo
        Hoooo = self.hbar.Hoooo
        Hvvvv = self.hbar.Hvvvv
        Hvovv = self.hbar.Hvovv
        Hooov = self.hbar.Hooov
        Hovvo = self.hbar.Hovvo
        Hvvvo = self.hbar.Hvvvo
        Hovoo = self.hbar.Hovoo

        lecc = self.pseudoenergy(o, v, ERI, l2)

        print("\nLCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E" % (0, lecc, -lecc))

        diis = helper_diis(l1, l2, max_diis)

        if self.ccwfn.model == 'CC3':
            Fov = Hov
            Woooo = self.ccwfn.build_Woooo_CC3(o, v, ERI, t1)
            Wovoo = self.ccwfn.build_Wovoo_CC3(o, v, ERI, t1, Woooo)
            Wooov = self.ccwfn.build_Wooov_CC3(o, v, ERI, t1)
            Wvovv = self.ccwfn.build_Wvovv_CC3(o, v, ERI, t1)
            Wvvvo = self.ccwfn.build_Wvvvo_CC3(o, v, ERI, t1)
            Wvvvv = self.build_Wvvvv_CC3(o, v, ERI, t1)
            Wovvo = self.build_Wovvo_CC3(o, v, ERI, t1)
            Zijal, Ziabd = self.CC3_noniter(o, v, t2, F, ERI, Wvvvo, Wovoo, self.ccwfn.t_alg)

        for niter in range(1, maxiter+1):
            lecc_last = lecc

            l1 = self.l1
            l2 = self.l2

            Goo = self.build_Goo(t2, l2)
            Gvv = self.build_Gvv(t2, l2)
            r1 = self.r_L1(o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)
            r2 = self.r_L2(o, v, l1, l2, ERI, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo)

            if self.ccwfn.model == 'CC3':
                if self.ccwfn.store_triples is True:
                    x1, x2 = self.CC3_iter_full(o, v, l1, l2, t2, F, ERI, Fov, Woooo, Wvvvv, Wvvvo, Wovoo, Wvovv, Wooov, Wovvo, Zijal, Ziabd)
                else:
                    x1, x2 = self.CC3_iter(o, v, l1, l2, t2, F, ERI, Fov, Woooo, Wvvvv, Wvvvo, Wovoo, Wvovv, Wooov, Wovvo, Zijal, Ziabd,
                            self.ccwfn.t_alg)
                r1 += x1; r2 += x2

            self.l1 += r1/Dia
            self.l2 += r2/Dijab
            rms = contract('ia,ia->', r1/Dia, r1/Dia)
            rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
            rms = np.sqrt(rms)

            lecc = self.pseudoenergy(o, v, ERI, self.l2)
            ediff = lecc - lecc_last
            print("LCC Iter %3d: LCC PseudoE = %.15f  dE = % .5E  rms = % .5E" % (niter, lecc, ediff, rms))

            if ((abs(ediff) < e_conv) and abs(rms) < r_conv):
                print("\nLambda-CC has converged in %.3f seconds." % (time.time() - lambda_tstart))
                print("\nLargest Lambda Amplitudes:")
                print_wfn(self.l1, self.l2)
                return lecc

            diis.add_error_vector(self.l1, self.l2)
            if niter >= start_diis:
                self.l1, self.l2 = diis.extrapolate(self.l1, self.l2)


    def build_Goo(self, t2, l2):
        return (1/2) * contract('mnef,inef->mi', t2, l2)


    def build_Gvv(self, t2, l2):
        return (-1/2) * contract('mnef,mnaf->ae', t2, l2)


    def r_L1(self, o, v, l1, l2, Hov, Hvv, Hoo, Hovvo, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        r_l1 = Hov.copy()
        r_l1 += contract('ie,ea->ia', l1, Hvv)
        r_l1 -= contract('ma,im->ia', l1, Hoo)
        r_l1 += (1/2) * contract('imef,efam->ia', l2, Hvvvo)
        r_l1 -= (1/2) * contract('mnae,iemn->ia', l2, Hovoo)
        r_l1 += contract('me,ieam->ia', l1, Hovvo)
        r_l1 -= contract('ef,eifa->ia', Gvv, Hvovv)
        r_l1 -= contract('mn,mina->ia', Goo, Hooov)
        return r_l1


    def r_L2(self, o, v, l1, l2, ERI, Hov, Hvv, Hoo, Hoooo, Hvvvv, Hovvo, Hvvvo, Hovoo, Hvovv, Hooov, Gvv, Goo):
        r_l2 = ERI[o,o,v,v].copy()
        r_l2 += contract('ia,jb->ijab', l1, Hov) - contract('ja,ib->ijab', l1, Hov)
        r_l2 += contract('jb,ia->ijab', l1, Hov) - contract('ib,ja->ijab', l1, Hov)
        r_l2 += contract('ijae,eb->ijab', l2, Hvv) - contract('ijbe,ea->ijab', l2, Hvv)
        r_l2 -= contract('imab,jm->ijab', l2, Hoo) - contract('jmab,im->ijab', l2, Hoo)
        r_l2 += 0.5 * contract('ijef,efab->ijab', l2, Hvvvv)
        r_l2 += 0.5 * contract('mnab,ijmn->ijab', l2, Hoooo)
        r_l2 += contract('ie,ejab->ijab', l1, Hvovv) - contract('je,eiab->ijab', l1, Hvovv)
        r_l2 -= contract('ma,ijmb->ijab', l1, Hooov) - contract('mb,ijma->ijab', l1, Hooov)
        tmp = contract('imae,jebm->ijab', l2, Hovvo)
        r_l2 += tmp - tmp.swapaxes(0,1) - tmp.swapaxes(2,3) + tmp.swapaxes(0,1).swapaxes(2,3)
        r_l2 += contract('be,ijae->ijab', Gvv, ERI[o,o,v,v]) - contract('ae,ijbe->ijab', Gvv, ERI[o,o,v,v])
        r_l2 -= contract('mj,imab->ijab', Goo, ERI[o,o,v,v]) - contract('mi,jmab->ijab', Goo, ERI[o,o,v,v])
        return r_l2


    def pseudoenergy(self, o, v, ERI, l2):
        return (1/4) * contract('ijab,ijab->', ERI[o,o,v,v], l2)


    # CC3 intermediates for <0|L2 [[H^,T3],nu1]|0> --> L1
    # These do not depend on L2, so we compute them once before the lambda iterations
    def CC3_noniter(self, o, v, t2, F, ERI, Wvvvo, Wovoo, alg='IJK'):
        no = t2.shape[0]
        nv = t2.shape[2]

        Zijal = np.zeros_like(ERI[o,o,v,o])
        Ziabd = np.zeros_like(ERI[o,v,v,v])

        if self.ccwfn.store_triples is True:
            t3 = self.ccwfn.t3
            Zijal = -(1/2) * contract('ijkabc,lkbc->ijal', t3, ERI[o,o,v,v])
            Ziabd = -(1/2) * contract('ijkabc,jkdc->iabd', t3, ERI[o,o,v,v])

        elif alg == 'IJK':
            for i in range(no):
                for j in range(no):
                    for k in range(no):
                        t3 = t3_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)
                        Zijal[i,j] -= (1/2) * contract('abc,lbc->al', t3, ERI[o,k,v,v])
                        Ziabd[i] -= (1/2) * contract('abc,dc->abd', t3, ERI[j,k,v,v])

        elif alg == 'AB':
            for a in range(nv):
                for b in range(nv):
                    t3 = t3_ab(o, v, a, b, t2, F, Wvvvo, Wovoo)
                    Zijal[:,:,a,:] -= (1/2) * contract('ijkc,lkc->ijl', t3, ERI[o,o,b+no,v])
                    Ziabd[:,a,b,:] -= (1/2) * contract('ijkc,jkdc->id', t3, ERI[o,o,v,v])

        return Zijal, Ziabd


    # CC3 intermediates and T3/L3 contributions to L1 and L2 equations
    # These must be computed in every lambda iteration
    def CC3_iter(self, o, v, l1, l2, t2, F, ERI, Fov, Woooo, Wvvvv, Wvvvo, Wovoo, Wvovv, Wooov, Wovvo, Zijal, Ziabd, alg='IJK'):
        no = l1.shape[0]
        nv = l1.shape[1]

        x2 = np.zeros_like(l2)

        # <0|L2 [[H^,T3],nu1]|0> --> L1
        Zia = np.zeros_like(l1)

        # <0|L3 [[H^,T2],nu1]|0> --> L1
        Ziabe = np.zeros((no,nv,nv,nv))
        Zijam = np.zeros((no,no,nv,no))
        Zjabd = np.zeros((no,nv,nv,nv))
        Zijlb = np.zeros((no,no,no,nv))

        if alg == 'IJK':
            # <0|L2 [[H^,T3],nu1]|0> --> L1
            for i in range(no):
                for j in range(no):
                    for k in range(no):
                        t3 = t3_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)
                        Zia[i] += (1/4) * contract('abc,bc->a', t3, l2[j,k])

            for i in range(no):
                for j in range(no):
                    for k in range(no):
                        l3 = l3_ijk(o, v, i, j, k, l1, l2, F, Fov, ERI[o,o,v,v], Wvovv, Wooov)

                        # <0|L3 [[H^,T2],nu1]|0> --> L1
                        Ziabe[i] += (1/2) * contract('abc,ec->abe', l3, t2[j,k])
                        Zijam[i,j] += (1/2) * contract('abc,mbc->am', l3, t2[o,k])
                        Zjabd[j] -= (1/2) * contract('abc,dc->abd', l3, t2[i,k])
                        Zijlb[i,j] -= (1/2) * contract('abc,lac->lb', l3, t2[o,k])

                        # <0|L3 [H^,nu2]|0> --> L2
                        x2[i,j] += (1/2) * contract('abc,bcd->ad', l3, Wvvvo[:,:,:,k])
                        x2[i,j] -= (1/2) * contract('dbc,bca->ad', l3, Wvvvo[:,:,:,k])
                        for l in range(no):
                            tmp = (1/2) * contract('abc,c->ab', l3, Wovoo[l,:,j,k])
                            x2[i,l] -= tmp
                            x2[l,i] += tmp

        elif alg == 'AB':
            for a in range(nv):
                for b in range(nv):
                    # <0|L2 [[H^,T3],nu1]|0> --> L1
                    t3 = t3_ab(o, v, a, b, t2, F, Wvvvo, Wovoo)
                    Zia[:,a] += (1/4) * contract('ijkc,jkc->i', t3, l2[:,:,b,:])

                    l3 = l3_ab(o, v, a, b, l1, l2, F, Fov, ERI[o,o,v,v], Wvovv, Wooov)

                    # <0|L3 [[H^,T2],nu1]|0> --> L1
                    Ziabe[:,a,b,:] += (1/2) * contract('ijkc,jkec->ie', l3, t2)
                    Zijam[:,:,a,:] += (1/2) * contract('ijkc,mkc->ijm', l3, t2[:,:,b,:])

#                    Zjabd[:,a,b,:] -= (1/2) * contract('ijkc,ikdc->jd', l3, t2)
#                    Zijlb[:,:,:,b] -= (1/2) * contract('ijkc,lkc->ijl', l3, t2[:,:,a,:])

                    # <0|L3 [H^,nu2]|0> --> L2
                    for d in range(nv):
                        tmp = (1/2) * contract('ijkc,ck->ij', l3, Wvvvo[b,:,d,:])
                        x2[:,:,a,d] += tmp
                        x2[:,:,d,a] -= tmp
                    x2[:,:,a,b] -= (1/2) * contract('ijkc,lcjk->il', l3, Wovoo)
                    x2[:,:,a,b] += (1/2) * contract('ljkc,icjk->il', l3, Wovoo)

        # <0|L2 [[H^,T3],nu1]|0> --> L1
        x1 = contract('ia,lida->ld', Zia, ERI[o,o,v,v])
        x1 += (1/2) * contract('ijal,ijad->ld', Zijal, l2)
        x1 += (1/2) * contract('iabd,ilab->ld', Ziabd, l2)

        # <0|L3 [[H^,T2],nu1]|0> --> L1
        x1 += (1/2) * contract('iabe,abde->id', Ziabe, Wvvvv)
        x1 += (1/2) * contract('ijam,lmij->la', Zijam, Woooo)
        x1 += contract('iabe,lbei->la', Ziabe, Wovvo)
        x1 += contract('ijam,madj->id', Zijam, Wovvo)

        return x1, x2


    def CC3_iter_full(self, o, v, l1, l2, t2, F, ERI, Fov, Woooo, Wvvvv, Wvvvo, Wovoo, Wvovv, Wooov, Wovvo, Zijal, Ziabd):
        tmp = contract('ia,jkbc->ijkabc', l1, ERI[o,o,v,v]) + contract('ia,jkbc->ijkabc', Fov, l2)
        l3 = permute_triples(tmp, 'i/jk', 'a/bc')

        tmp = contract('ijad,dkbc->ijkabc', l2, Wvovv)
        l3 += permute_triples(tmp, 'k/ij', 'a/bc')

        tmp = -contract('ilab,jklc->ijkabc', l2, Wooov)
        l3 += permute_triples(tmp, 'i/jk', 'c/ab')

        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        denom = occ.reshape(-1,1,1,1,1,1) + occ.reshape(-1,1,1,1,1) + occ.reshape(-1,1,1,1) - vir.reshape(-1,1,1) - vir.reshape(-1,1) - vir
        l3 = l3/denom

        # Save the L3s for later use (e.g., in the CC3 response codes)
        self.l3 = l3

        # <0|L2[[H^,T3],nu1]|0> -> L1
        x1 = np.zeros_like(l1)
        t3 = self.ccwfn.t3
        tmp = (1/4) * contract('ijkabc,jkbc->ia', t3, l2)
        x1 = contract('ia,lida->ld', tmp, ERI[o,o,v,v])
        x1 += (1/2) * contract('ijal,ijad->ld', Zijal, l2)
        x1 += (1/2) * contract('iabd,ilab->ld', Ziabd, l2)

        # <0|L3[[H^,T2],nu1]|0> -> L1
        tmp = (1/2) * contract('ijkabc,jkec->iabe', l3, t2)
        x1 += (1/2) * contract('iabe,abde->id', tmp, Wvvvv)
        tmp = (1/2) * contract('ijkabc,mkbc->ijam', l3, t2)
        x1 += (1/2) * contract('ijam,lmij->la', tmp, Woooo)
        tmp = -(1/2) * contract('ijkabc,ikdc->jabd', l3, t2)
        x1 += contract('jabd,lbdj->la', tmp, Wovvo)
        tmp = -(1/2) * contract('ijkabc,lkac->ijlb', l3, t2)
        x1 += contract('ijlb,lbdj->id', tmp, Wovvo)

        # <0|L3[H^,nu2]|0> -> L2
        tmp = (1/2) * contract('ijkabc,bcdk->ijad', l3, Wvvvo)
        x2 = tmp - tmp.swapaxes(2,3)
        tmp = -(1/2) * contract('ijkabc,lcjk->ilab', l3, Wovoo)
        x2 += tmp - tmp.swapaxes(0,1)

        return x1, x2


    def build_Wvvvv_CC3(self, o, v, ERI, t1):
        Wvvvv = ERI[v,v,v,v].copy()
        Wvvvv -= contract('mb,amef->abef', t1, ERI[v,o,v,v]) - contract('ma,bmef->abef', t1, ERI[v,o,v,v])
        tau = contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        Wvvvv += (1/2) * contract('mnab,mnef->abef', tau, ERI[o,o,v,v])
        return Wvvvv


    def build_Wovvo_CC3(self, o, v, ERI, t1):
        Wovvo = ERI[o,v,v,o].copy()
        Wovvo += contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
        Wovvo -= contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
        Wovvo -= contract('jf,nb,mnef->mbej', t1, t1, ERI[o,o,v,v])
        return Wovvo
