"""
ccresponse.py: CC Response Functions
"""

import numpy as np
import time
from opt_einsum import contract
from .utils import helper_diis, print_wfn, permute_triples
from .cctriples import t3c_ijk, t3c_abc, l3_ijk, l3_abc, X3_ijk, X3_abc

class ccresponse(object):
    """
    An spin-orbital CC Response Property Object.

    Methods
    -------
    linresp():
        Compute a CC linear response function.
    solve_right():
        Solve the right-hand perturbed wave function equations.
    pertcheck():
        Check first-order perturbed wave functions for all available perturbation operators.
    """

    def __init__(self, ccwfn, cclambda):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            Contains the CC T1 and T2 amplitudes, as well as the Hamiltonian integrals
        cclambda : PyCC cclambda object
            Contains the CC L1 and L2 amplitudes, as well as the similarity-transformed Hamiltonian
        omega1 : scalar
            The first external field frequency (for linear and quadratic response functions)
        omega2 : scalar
            The second external field frequency (for quadratic response functions, not yet implemented)

        Returns
        -------
        None
        """

        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.H = ccwfn.H
        self.hbar = cclambda.hbar

        # Cartesian indices
        self.cart = ["X", "Y", "Z"]

        # HBAR-based denominators
        eps_occ = np.diag(self.hbar.Hoo)
        eps_vir = np.diag(self.hbar.Hvv)
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir


    def polarizability(self, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
        """
        Computes the dipole polarizability in the length gauge at energy omega (w, au):

        alpha_w = -<<mu;mu>>_w = - <0|(1+L) { [muBAR,X(mu,-w)] + [muBAR, X(mu,w) + [[HBAR,X(mu,-w)],X(mu,w)] } |0>

        """

        # Build dictionary of similarity-transformed dipole integrals
        self.pertbar = {}

        # Electric-dipole operator (length)
        for axis in range(2,3):
            key = "MU_" + self.cart[axis]
            self.pertbar[key] = pertbar(self.H.mu[axis], self.ccwfn)

        # Build the perturbed wave functions, stored in a dictionary of lists
        X = {}
        for axis in range(2,3):
            # A(w) or A(0)
            pertkey = "MU_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)

            # A(-w)
            if (omega != 0.0):
                pertkey = "MU_" + self.cart[axis]
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)

        polar = np.zeros((3,3))
        for alpha in range(2,3):
            pertkey = "MU_" + self.cart[alpha]
            if omega != 0.0:
                X_key = pertkey + "_" + f"{-omega:0.6f}"
            else:
                X_key = pertkey + "_" + f"{omega:0.6f}"
            pert_A = self.pertbar[pertkey]; X_A = X[X_key]
            for beta in range(2,3):
                pertkey = "MU_" + self.cart[beta]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                pert_B = self.pertbar[pertkey]; X_B = X[X_key]

                components = self.linresp(pert_A, X_A, pert_B, X_B)
                print(alpha, beta)
                print("LCX    = ", f"{components[0]:20.15f}")
                print("LHX1Y1 = ", f"{components[1]+components[2]:20.15f}")
                print("LHX1Y2 = ", f"{components[4]+components[5]:20.15f}")
                print("LHX2Y2 = ", f"{components[3]:20.15f}")
                if self.ccwfn.model == 'CC3':
                    print("CC3 Contributions:")
                    print("L2CX3     = ", f"{components[6]:20.15f}")
                    print("L3CX3     = ", f"{components[7]:20.15f}")
                    print("L3CX1T3   = ", f"{components[8]:20.15f}")
                    print("L3CX2T2   = ", f"{components[9]:20.15f}")
                    print("L2HX1Y3   = ", f"{components[10]:20.15f}")
                    print("L3HX1Y2   = ", f"{components[11]:20.15f}")
                    print("L3HX1Y1T2 = ", f"{components[12]:20.15f}")

                polar[alpha,beta] = -1.0 * np.sum(components)

        print(f"{self.ccwfn.model:s} Polarizability Tensor (Length Gauge):")
        print(polar)
        print(f"Evaluated at omega = {omega:8.6f} E_h")

        return polar

    def optrot(self, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
        """
        Computes the dipole polarizability in the length gauge at energy omega (w, au):

        G'_w = << mu ; m >>_w = (1/2) <0|(1+L) { [muBAR, X(m,w)] + [mBAR, X(mu,-w) + [[HBAR,X(mu,-w)],X(m,w)] } |0>
                              + (1/2) <0|(1+L) { [muBAR, X(m*,-w)] + [m*BAR, X(mu,w) + [[HBAR, X(m*,-w)], X(mu,w)] }|0>
        """

        if omega == 0.0:
            raise Exception("The field frequency cannot be zero for optical rotations.")

        # Build dictionary of required similarity-transformed property integrals
        self.pertbar = {}

        # Electric-dipole operator (length)
        for axis in range(3):
            key = "MU_" + self.cart[axis]
            self.pertbar[key] = pertbar(self.H.mu[axis], self.ccwfn)

        # Magnetic-dipole operator
        for axis in range(3):
            key = "M_" + self.cart[axis]
            self.pertbar[key] = pertbar(self.H.m[axis], self.ccwfn)

        # Complex-conjugate of magnetic-dipole operator
        for axis in range(3):
            key = "M*_" + self.cart[axis]
            self.pertbar[key] = pertbar(-1.0*self.H.m[axis], self.ccwfn)

        # Build the perturbed wave functions, stored in dictionaries
        X = {}
        for axis in range(3):
            # Mu(-w)
            pertkey = "MU_" + self.cart[axis]
            X_key = pertkey + "_" + f"{-omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)

            # M(w)
            pertkey = "M_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)

            # Mu(w)
            pertkey = "MU_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)

            # M*(-w)
            pertkey = "M*_" + self.cart[axis]
            X_key = pertkey + "_" + f"{-omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)

        tensor = np.zeros((3,3))
        for alpha in range(3):
            pertkey = "MU_" + self.cart[alpha]
            X_key = pertkey + "_" + f"{-omega:0.6f}"
            pert_A = self.pertbar[pertkey]; X_A = X[X_key]
            for beta in range(3):
                pertkey = "M_" + self.cart[beta]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                pert_B = self.pertbar[pertkey]; X_B = X[X_key]

                polar = self.linresp(pert_A, X_A, pert_B, X_B)
                print(alpha, beta)
                print("LCX    = ", f"{polar[0]:20.15f}")
                print("LHX1Y1 = ", f"{polar[1]+polar[2]:20.15f}")
                print("LHX1Y2 = ", f"{polar[4]+polar[5]:20.15f}")
                print("LHX2Y2 = ", f"{polar[3]:20.15f}")

                tensor[alpha,beta] = (1/2) * sum(polar)

        for alpha in range(3):
            pertkey = "MU_" + self.cart[alpha]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            pert_A = self.pertbar[pertkey]; X_A = X[X_key]
            for beta in range(3):
                pertkey = "M*_" + self.cart[beta]
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                pert_B = self.pertbar[pertkey]; X_B = X[X_key]

                polar = self.linresp(pert_A, X_A, pert_B, X_B)
                print(alpha, beta)
                print("LCX    = ", f"{polar[0]:20.15f}")
                print("LHX1Y1 = ", f"{polar[1]+polar[2]:20.15f}")
                print("LHX1Y2 = ", f"{polar[4]+polar[5]:20.15f}")
                print("LHX2Y2 = ", f"{polar[3]:20.15f}")

                tensor[alpha,beta] += (1/2) * sum(polar)

        print(f"{self.ccwfn.model:s} Optical Rotation Tensor (Length Gauge):")
        print(tensor)
        print(f"Evaluated at omega = {omega:8.6f} E_h")

        return tensor


    def linresp(self, A, X_A, B, X_B):
        """
        Calculate half of the CC linear-response function for one-electron perturbations A and B at frequency omega (w):

            <<A;B>>_w = (1/2) <0|(1+L) { [ABAR,X(B,w)] + [BBAR, X(A,-w)]  + [[HBAR,X(A,-w)],X(B,w)]} |0>

        The other half of the response function using the complex conjugates of the operators and the swapped
        frequencies must be generated by a separate call of this function.

        Note that this is for specific cartesian components of the perturbations, and both the perturbed wave 
        functions and the similarity-transformed operators must be computed outside this function.

        Parameters:
        -----------
        A: pertbar object
            pertbar object associated with the left-hand perturbation
        X_A: List of Numpy arrays
            List of singles, doubles, etc. arrays of the perturbed wave function for the left-hand perturbation
        B: pertbar object
            pertbar object associated with right-hand perturbation
        X_B: List of Numpy arrays
            List of singles, doubles, etc. arrays of the perturbed wave function for the right-hand perturbation

        Returns:
        --------
        linresp: float
            The value of the specific component of the linear response tensor requested
        """

        o = self.ccwfn.o
        v = self.ccwfn.v
        ERI = self.H.ERI

        #<0|(1+L) [ABAR, X_B]|0>
        polar_LCX = self.LCX(A, X_B)
        polar_LCX += self.LCX(B, X_A)

        #<0|[[HBAR, X1_A], X1_B]|0>
        polar_HXY = contract('ijab,ia,jb->', ERI[o,o,v,v], X_A[1], X_B[1])

        #<0|L[[HBAR, X1_A], X1_B]|0>
        polar_LHX1Y1 = self.LHX1Y1(X_A, X_B)

        #<0|L[[HBAR, X2_A], X2_B]|0>
        polar_LHX2Y2 = self.LHX2Y2(X_A, X_B)

        #<0|L[[HBAR, X1_A], X2_B]|0>
        polar_LHX1Y2 = self.LHX1Y2(X_A, X_B)

        #<0|L[[HBAR, X1_B], X2_A]|0>
        polar_LHY1X2 = self.LHX1Y2(X_B, X_A)

        polar = np.array([polar_LCX, polar_HXY, polar_LHX1Y1, polar_LHX2Y2, polar_LHX1Y2, polar_LHY1X2])

        if self.ccwfn.model == 'CC3':
            if self.ccwfn.store_triples is True:
                #<0|L2[C,X3]|0> + <0|L3[C^,X3]|0> + <0|L3[[C,X1],T3]|0> + <0|L3[[C,X2],T2]|0>
                polar_LCX_CC3 = self.LCX_CC3(A, X_B)
                polar_LCX_CC3 += self.LCX_CC3(B, X_A)

                #<0|L2[[H,X1],Y3]|0>
                polar_L2HX1Y3 = self.L2HX1Y3_CC3(X_A, X_B)
                polar_L2HX1Y3 += self.L2HX1Y3_CC3(X_B, X_A)

                #<0|L3[[H^,X1],Y2]|0>
                polar_L3HX1Y2 = self.L3HX1Y2_CC3(X_A, X_B)
                polar_L3HX1Y2 += self.L3HX1Y2_CC3(X_B, X_A)

                #<0|L3[[[H^,X1],Y1],T2]|0>
                polar_L3HX1Y1T2 = self.L3HX1Y1T2_CC3(X_A, X_B)

            else:
                t1 = self.ccwfn.t1
                t2 = self.ccwfn.t2
                l1 = self.cclambda.l1
                l2 = self.cclambda.l2
                F = self.ccwfn.H.F

                omega_A = X_A[0]
                omega_B = X_B[0]
                X1_A = X_A[1]
                X1_B = X_B[1]
                X2_A = X_A[2]
                X2_B = X_B[2]

                Zvvvo_A, Zovoo_A = self.CC3_linresp_intermediates(o, v, A, t1, t2, X_A[1], ERI)
                Zvvvo_B, Zovoo_B = self.CC3_linresp_intermediates(o, v, B, t1, t2, X_B[1], ERI)

                Fov = self.hbar.Hov
                Wvovv = self.hbar.Hvovv
                Wooov = self.hbar.Hooov
                Woooo = self.ccwfn.build_Woooo_CC3(o, v, ERI, t1)
                Wvvvo = self.ccwfn.build_Wvvvo_CC3(o, v, ERI, t1)
                Wovoo = self.ccwfn.build_Wovoo_CC3(o, v, ERI, t1, Woooo)

                # For <0|L2[[H,X1],Y3]|0>
                Zia_A = contract('ld,lida->ia', X1_A, ERI[o,o,v,v])
                Zia_B = contract('ld,lida->ia', X1_B, ERI[o,o,v,v])
                Zijlb_A = contract('ld,ijdb->ijlb', X1_A, ERI[o,o,v,v])
                Zijlb_B = contract('ld,ijdb->ijlb', X1_B, ERI[o,o,v,v])
                Zdjab_A = -contract('ld,ljab->djab', X1_A, ERI[o,o,v,v])
                Zdjab_B = -contract('ld,ljab->djab', X1_B, ERI[o,o,v,v])

                polar_L2CX3 = 0.0
                polar_L3CX3 = 0.0
                polar_L3CX1T3 = 0.0
                polar_L3CX2T2 = 0.0
                polar_L2HX1Y3 = 0.0
                polar_L3HX1Y2 = 0.0
                polar_L3HX1Y1T2 = 0.0

                no = self.ccwfn.no
                Zmk_A = np.zeros((no,no))
                Zmk_B = np.zeros((no,no))
                for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            X3_A = X3_ijk(o, v, i, j, k, t2, F, A, X2_A, Wvvvo, Wovoo, Zvvvo_A, Zovoo_A, omega_A)
                            X3_B = X3_ijk(o, v, i, j, k, t2, F, B, X2_B, Wvvvo, Wovoo, Zvvvo_B, Zovoo_B, omega_B)
                            l3 = l3_ijk(o, v, i, j, k, l1, l2, F, Fov, ERI[o,o,v,v], Wvovv, Wooov)
                            t3 = t3c_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)

                            # <0|L2[A,X3]|0>
                            tmp = contract('e,abe->ab', A.Aov[k], X3_B)
                            polar_L2CX3 += (1/4) * contract('ab,ab->', l2[i,j], tmp)
                            tmp = contract('e,abe->ab', B.Aov[k], X3_A)
                            polar_L2CX3 += (1/4) * contract('ab,ab->', l2[i,j], tmp)

                            # <0|L3[A^,X3]|0>
                            tmp = contract('abe,ce->abc', X3_A, B.Avv)
                            tmp += contract('abe,ce->abc', X3_B, A.Avv)
                            polar_L3CX3 += (1/12) * contract('abc,abc->', tmp, l3)
                            tmp = -(1/12) * contract('abc,abc->', l3, X3_A)
                            for m in range(no):
                                Zmk_A[m,k] += tmp
                            tmp = -(1/12) * contract('abc,abc->', l3, X3_B)
                            for m in range(no):
                                Zmk_B[m,k] += tmp

                            # <0|L2[[H,X1],Y3]|0>
                            tmp = contract('abc,a->bc', X3_A, Zia_B[i])
                            tmp += contract('abc,a->bc', X3_B, Zia_A[i])
                            polar_L2HX1Y3 += (1/4) * contract('bc,bc->', tmp, l2[j,k])
#                            for l in range(no):
#                                tmp = (1/2) * contract('abc,b->ac', X3_A, Zijlb_B[i,j,l])
#                                tmp += (1/2) * contract('abc,b->ac', X3_B, Zijlb_A[i,j,l])
#                                polar_L2HX1Y3 -= (1/2) * contract('ac,ac->', tmp, l2[l,k])
#                            tmp = (1/2) * contract('abc,dab->dc', X3_A, Zdjab_B[:,j])
#                            tmp += (1/2) * contract('abc,dab->dc', X3_B, Zdjab_A[:,j])
#                            polar_L2HX1Y3 += (1/2) * contract('dc,dc->', tmp, l2[i,k])

                nv = self.ccwfn.nv
                Zce_A = np.zeros((nv,nv))
                Zce_B = np.zeros((nv,nv))
                for a in range(nv):
                    for b in range(nv):
                        for c in range(nv):
                            X3_A = X3_abc(o, v, a, b, c, t2, F, A, Wvvvo, Wovoo, omega_A)
                            X3_B = X3_abc(o, v, a, b, c, t2, F, B, Wvvvo, Wovoo, omega_B)
                            l3 = l3_abc(o, v, a, b, c, l1, l2, F, Fov, ERI[o,o,v,v], Wvovv, Wooov)
                            t3 = t3c_abc(o, v, a, b, c, t2, F, Wvvvo, Wovoo)

                            # <0|L2[A,X3]|0>
                            tmp = contract('m,ijm->ij', A.Aov[:,c], X3_B)
                            polar_L2CX3 += (1/4) * contract('ij,ij->', l2[:,:,a,b], tmp)
                            tmp = contract('m,ijm->ij', B.Aov[:,c], X3_A)
                            polar_L2CX3 += (1/4) * contract('ij,ij->', l2[:,:,a,b], tmp)

                            # <0|L3[A^,X3]|0>
                            tmp = (1/12) * contract('ijk,ijk->', l3, X3_A)
                            for e in range(nv):
                                Zce_A[c,e] += tmp
                            tmp = (1/12) * contract('ijk,ijk->', l3, X3_B)
                            for e in range(nv):
                                Zce_B[c,e] += tmp
                            tmp = -contract('ijm,mk->ijk', X3_A, B.Aoo)
                            tmp -= contract('ijm,mk->ijk', X3_B, A.Aoo)
                            polar_L3CX3 += (1/12) * contract('ijk,ijk->', tmp, l3)

                            # <0|L2[[H,X1],Y3]|0>
                            tmp = contract('ijk,i->jk', X3_A, Zia_B[:,a])
                            tmp += contract('ijk,i->jk', X3_B, Zia_A[:,a])
                            polar_L2HX1Y3 += (1/4) * contract('jk,jk->', tmp, l2[:,:,b,c])
#                            tmp = (1/2) * contract('ijk,ijl->kl', X3_A, Zijlb_B[:,:,:,b])
#                            tmp += (1/2) * contract('ijk,ijl->kl', X3_B, Zijlb_A[:,:,:,b])
#                            polar_L2HX1Y3 -= (1/2) * contract('kl,kl->', tmp, l2[:,:,a,c])
#                            for d in range(nv):
#                                tmp = (1/2) * contract('ijk,j->ik', X3_A, Zdjab_B[d,:,a,b])
#                                tmp += (1/2) * contract('ijk,j->ik', X3_B, Zdjab_A[d,:,a,b])
#                                polar_L2HX1Y3 += (1/2) * contract('ik,ik->', tmp, l2[:,:,d,c])

                # Remainder of <0|L3[A^,X3]|0>
                polar_L3CX3 += contract('ce,ce->', Zce_A, B.Avv)
                polar_L3CX3 += contract('ce,ce->', Zce_B, A.Avv)
                polar_L3CX3 += contract('mk,mk->', Zmk_A, B.Aoo)
                polar_L3CX3 += contract('mk,mk->', Zmk_B, A.Aoo)

                polar_LCX_CC3 = np.array([polar_L2CX3, polar_L3CX3, polar_L3CX1T3, polar_L3CX2T2])

            polar = np.append(polar, polar_LCX_CC3)
            polar = np.append(polar, [polar_L2HX1Y3, polar_L3HX1Y2, polar_L3HX1Y1T2])

        return polar


    def LCX(self, pert, X):
        o = self.ccwfn.o
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2

        X1 = X[1]
        X2 = X[2]

        polar = contract('ia,ia->', pert.Aov, X1) # diagram 1

        tmp = contract('ae,ie->ia', pert.Avv, X1) # diagram 2
        tmp -= contract('mi,ma->ia', pert.Aoo, X1) # diagram 3
        tmp += contract('me,imae->ia', pert.Aov, X2) # diagram 8
        polar += contract('ia,ia->', l1, tmp)

        tmp = contract('abej,ie->ijab', pert.Avvvo, X1) # diagram 4
        tmp -= contract('mbij,ma->ijab', pert.Aovoo, X1) # diagram 5
        tmp += contract('be,ijae->ijab', pert.Avv, X2) # diagram 6
        tmp -= contract('mj,imab->ijab', pert.Aoo, X2) # diagram 7
        polar += (1/2) * contract('ijab,ijab->', l2, tmp)

        return polar


    def LHX1Y1(self, X, Y):
        o = self.ccwfn.o
        v = self.ccwfn.v
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        ERI = self.H.ERI

        X1 = X[1]
        Y1 = Y[1]

        tau = contract('ia,jb->ijab', X1, Y1) + contract('ia,jb->ijab', Y1, X1)

        tmp = -1.0 * contract('me,imea->ia', hbar.Hov, tau) #diagrams 1 and 2
        tmp += contract('amef,imef->ia', hbar.Hvovv, tau) #diagrams 7 and 8
        tmp -= contract('mnie,mnae->ia', hbar.Hooov, tau) #diagrams 9 and 10
        polar = contract('ia,ia->', l1, tmp)

        Zvv = (1/2) * contract('mnef,mneb->fb', ERI[o,o,v,v], tau)
        Zoo = (1/2) * contract('mnef,mjef->nj', ERI[o,o,v,v], tau)

        tmp = (1/2) * contract('mnij,ma,nb->ijab', hbar.Hoooo, X1, Y1) # diagram 3
        tmp += (1/2) * contract('abef,ie,jf->ijab', hbar.Hvvvv, X1, Y1) # diagram 4
        tmp -= contract('mbej,imea->ijab', hbar.Hovvo, tau) # diagrams 5 and 6
        tmp -= contract('ijaf,fb->ijab', t2, Zvv) # diagrams 11 and 12
        tmp -= contract('inab,nj->ijab', t2, Zoo) # diagrams 13 and 14
        polar += contract('ijab,ijab->', l2, tmp)

        return polar


    def LHX2Y2(self, X, Y):
        o = self.ccwfn.o
        v = self.ccwfn.v
        t2 = self.ccwfn.t2
        l2 = self.cclambda.l2
        ERI = self.H.ERI

        X2 = X[2]
        Y2 = Y[2]

        Zovvo = contract('mnef,njfb->mbej', ERI[o,o,v,v], Y2)
        Zoooo_A = (1/4) * contract('mnef,ijef->mnij', ERI[o,o,v,v], X2)
        Zoooo_B = (1/4) * contract('mnef,ijef->mnij', ERI[o,o,v,v], Y2)
        Zvv_A = (-1/2) * contract('mnef,mnbf->eb', ERI[o,o,v,v], Y2)
        Zvv_B = (-1/2) * contract('mnef,mnbf->eb', ERI[o,o,v,v], X2)
        Zoo_A = (-1/2) * contract('mnef,jnef->mj', ERI[o,o,v,v], Y2)
        Zoo_B = (-1/2) * contract('mnef,jnef->mj', ERI[o,o,v,v], X2)

        tmp = contract('mbej,imae->ijab', Zovvo, X2) # diagram 1
        tmp += (1/4) * contract('mnij,mnab->ijab', Zoooo_A, Y2) # diagram 2
        tmp += (1/4) * contract('mnij,mnab->ijab', Zoooo_B, X2) # diagram 3
        tmp += (1/2) * contract('eb,ijae->ijab', Zvv_A, X2) # diagram 4
        tmp += (1/2) * contract('eb,ijae->ijab', Zvv_B, Y2) # diagram 6
        tmp += (1/2) * contract('mj,imab->ijab', Zoo_A, X2) # diagram 5
        tmp += (1/2) * contract('mj,imab->ijab', Zoo_B, Y2) # diagram 7

        polar = contract('ijab,ijab->', l2, tmp)

        return polar


    def LHX1Y2(self, X, Y):
        o = self.ccwfn.o
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        ERI = self.H.ERI

        X1 = X[1]
        Y2 = Y[2]

        Zov = contract('mnef,me->nf', ERI[o,o,v,v], X1)
        Zvv = (-1/2) * contract('mnef,mnaf->ea', ERI[o,o,v,v], Y2)
        Zoo = (-1/2) * contract('mnef,inef->mi', ERI[o,o,v,v], Y2)
        tmp = contract('nf,nifa->ia', Zov, Y2) # diagram 3
        tmp += contract('ea,ie->ia', Zvv, X1) # diagram 4
        tmp += contract('mi,ma->ia', Zoo, X1) # diagram 5

        polar = contract('ia,ia->', l1, tmp)

        Zoo = contract('me,ie->mi', hbar.Hov, X1)
        Zoo += contract('mnie,ne->mi', hbar.Hooov, X1)
        Zvv = contract('me,ma->ea', hbar.Hov, X1)
        Zvv -= contract('amef,mf->ea', hbar.Hvovv, X1)
        Zvoov = contract('anfe,if->anie', hbar.Hvovv, X1)
        Zvoov -= contract('mnie,ma->anie', hbar.Hooov, X1)
        Zoooo = (1/2) * contract('mnie,je->mnij', hbar.Hooov, X1)
        Zoovo = (1/2) * contract('amef,ijef->ijam', hbar.Hvovv, Y2)

        tmp = (-1/2) * contract('mi,mjab->ijab', Zoo, Y2) # diagrams 1 and 7
        tmp -= (1/2) * contract('ea,ijeb->ijab', Zvv, Y2) # diagrams 2 and 9
        tmp += contract('anie,njeb->ijab', Zvoov, Y2) # diagrams 6 and 8
        tmp += (1/2) * contract('mnij,mnab->ijab', Zoooo, Y2) # diagram 10
        tmp -= (1/2) * contract('ijam,mb->ijab', Zoovo, X1) # diagram 11

        polar += contract('ijab,ijab->', l2, tmp)

        return polar


    def LCX_CC3(self, pert, X):
        o = self.ccwfn.o
        v = self.ccwfn.v

        t2 = self.ccwfn.t2
        l2 = self.cclambda.l2
        X1 = X[1]
        X2 = X[2]

        t3 = self.ccwfn.t3
        l3 = self.cclambda.l3
        X3 = X[3]

        # <0|L2[C,X3]|0>
        tmp = (1/4) * contract('ijab,ijkabc->kc', l2, X3)
        polar_L2CX3 = contract('kc,kc->', tmp, pert.Aov)

        # <0|L3[C^,X3]|0>
        tmp = contract('ijkabc,ijkabe->ce', l3, X3)
        polar_L3CX3 = (1/12) * contract('ce,ce->', tmp, pert.Avv)
        tmp = contract('ijkabc,ijmabc->mk', l3, X3)
        polar_L3CX3 -= (1/12) * contract('mk,mk->', tmp, pert.Aoo)

        # <0|L3[[C,X1],T3]|0>
        tmp1 = contract('mc,me->ce', X1, pert.Aov)
        tmp2 = -(1/12) * contract('ijkabc,ijkabe->ce', l3, t3)
        polar_L3CX1T3 = contract('ce,ce->', tmp1, tmp2)
        tmp1 = contract('ke,me->mk', X1, pert.Aov)
        tmp2 = -(1/12) * contract('ijkabc,ijmabc->mk', l3, t3)
        polar_L3CX1T3 += contract('mk,mk->', tmp1, tmp2)

        # <0|L3[[C,X2],T2]|0>
        tmp = (1/2) * contract('ijkabc,mkbc->ijam', l3, t2)
        tmp = -(1/2) * contract('ijam,ijae->me', tmp, X2)
        polar_L3CX2T2 = contract('me,me->', tmp, pert.Aov)
        tmp = (1/2) * contract('ijkabc,imab->jkcm', l3, X2)
        tmp = -(1/2) * contract('jkcm,jkec->me', tmp, t2)
        polar_L3CX2T2 += contract('me,me->', tmp, pert.Aov)

        return np.array([polar_L2CX3, polar_L3CX3, polar_L3CX1T3, polar_L3CX2T2])


    def L2HX1Y3_CC3(self, X, Y):
        o = self.ccwfn.o
        v = self.ccwfn.v

        ERI = self.H.ERI
        l2 = self.cclambda.l2
        X1 = X[1]
        Y3 = Y[3]

        tmp = contract('me,mnef->nf', X1, ERI[o,o,v,v])
        tmp = contract('nijfab,nf->ijab', Y3, tmp)
        polar_L2HX1Y3 = (1/4) * contract('ijab,ijab->', l2, tmp)

#        tmp = contract('ie,mnef->mnif', X1, ERI[o,o,v,v])
#        tmp = contract('mnjafb,mnif->ijab', Y3, tmp)
#        polar_L2HX1Y3 -= (1/4) * contract('ijab,ijab->', l2, tmp)

#        tmp = contract('ma,mnef->anef', X1, ERI[o,o,v,v])
#        tmp = contract('injefb,anef->ijab', Y3, tmp)
#        polar_L2HX1Y3 -= (1/4) * contract('ijab,ijab->', l2, tmp)

        return polar_L2HX1Y3


    def L3HX1Y2_CC3(self, X, Y):
        o = self.ccwfn.o
        v = self.ccwfn.v

        ERI = self.H.ERI
        t1 = self.ccwfn.t1
        l3 = self.cclambda.l3

        Woooo = self.ccwfn.build_Woooo_CC3(o, v, ERI, t1)
        Wvvvv = self.cclambda.build_Wvvvv_CC3(o, v, ERI, t1)
        Wovvo = self.cclambda.build_Wovvo_CC3(o, v, ERI, t1)

        X1 = X[1]
        Y2 = Y[2]

        tmp = contract('ie,abef->abif', X1, Wvvvv)
        tmp = contract('ijkabc,abif->jkfc', l3, tmp)
        polar_L3HX1Y2 = (1/4) * contract('jkfc,jkfc->', Y2, tmp)

        tmp = contract('ma,mnij->anij', X1, Woooo)
        tmp = contract('ijkabc,anij->nkbc', l3, tmp)
        polar_L3HX1Y2 += (1/4) * contract('nkbc,nkbc->', Y2, tmp)

        tmp = contract('ie,mbej->mbij', X1, Wovvo)
        tmp = contract('ijkabc,mbij->mkac', l3, tmp)
        polar_L3HX1Y2 -= (1/2) * contract('mkac,mkac->', Y2, tmp)

        tmp = contract('ma,mbej->abej', X1, Wovvo)
        tmp = contract('ijkabc,abej->ikec', l3, tmp)
        polar_L3HX1Y2 -= (1/2) * contract('ikec,ikec->', Y2, tmp)

        return polar_L3HX1Y2


    def L3HX1Y1T2_CC3(self, X, Y):
        o = self.ccwfn.o
        v = self.ccwfn.v

        ERI = self.H.ERI
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        l3 = self.cclambda.l3

        Wooov = self.ccwfn.build_Wooov_CC3(o, v, ERI, t1)
        Wvovv = self.ccwfn.build_Wvovv_CC3(o, v, ERI, t1)

        X1 = X[1]
        Y1 = Y[1]

        tau = contract('ia,jb->ijab', X1, Y1) + contract('jb,ia->ijab', X1, Y1)

        tmp = contract('kmfc,bmef->bcek', tau, Wvovv)
        tmp = contract('ijkabc,bcek->ijae', l3, tmp)
        polar_L3HX1Y1T2 = -(1/2) * contract('ijae,ijae->', t2, tmp)

        tmp = contract('ie,jf,amef->amij', X1, Y1, Wvovv)
        tmp = contract('ijkabc,amij->mkbc', l3, tmp)
        polar_L3HX1Y1T2 -= (1/2) * contract('mkbc,mkbc->', t2, tmp)

        tmp = contract('knec,mnje->mcjk', tau, Wooov)
        tmp = contract('ijkabc,mcjk->imab', l3, tmp)
        polar_L3HX1Y1T2 += (1/2) * contract('imab,imab->', t2, tmp)

        tmp = contract('ma,nb,mnie->abie', X1, Y1, Wooov)
        tmp = contract('ijkabc,abie->jkec', l3, tmp)
        polar_L3HX1Y1T2 += (1/2) * contract('jkec,jkec->', t2, tmp)

        return polar_L3HX1Y1T2


    def solve_right(self, pertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
        solver_start = time.time()

        store_triples = self.ccwfn.store_triples

        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.H.F
        ERI = self.H.ERI
        t1 = self.ccwfn.t1
        t2 = self.ccwfn.t2
        Dia = self.Dia
        Dijab = self.Dijab

        # initial guess
        X1 = pertbar.Avo.T/(Dia + omega)
        X2 = pertbar.Avvoo/(Dijab + omega)

        pseudo = self.pseudoresponse(pertbar, X1, X2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        diis = helper_diis(X1, X2, max_diis)

        self.X1 = X1
        self.X2 = X2

        if self.ccwfn.model == 'CC3':
            Fov = self.hbar.Hov
            Woooo = self.ccwfn.build_Woooo_CC3(o, v, ERI, t1)
            Wovoo = self.ccwfn.build_Wovoo_CC3(o, v, ERI, t1, Woooo)
            Wvvvo = self.ccwfn.build_Wvvvo_CC3(o, v, ERI, t1)
            Wvvvv = self.cclambda.build_Wvvvv_CC3(o, v, ERI, t1)
            Wovvo = self.cclambda.build_Wovvo_CC3(o, v, ERI, t1)
            if store_triples is False:
                Yoovo, Yovvv, Zovoo, Zvvvo = self.CC3_noniter(o, v, t2, F, ERI, Wvvvo, Wovoo, pertbar)

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            X1 = self.X1
            X2 = self.X2

            r1 = self.r_X1(pertbar, omega)
            r2 = self.r_X2(pertbar, omega)

            if self.ccwfn.model == 'CC3':
                if store_triples is True:
                    z1, z2 = self.CC3_iter_full(o, v, X1, X2, t2, F, ERI, self.hbar, pertbar, Wvvvv, Woooo, Wovvo, Wvvvo, Wovoo, omega)
                else:
                    z1, z2 = self.CC3_iter(o, v, X1, X2, t2, F, ERI, self.hbar, pertbar, Fov, Wvvvv, Woooo, Wovvo, Wvvvo, Wovoo,
                    Yoovo, Yovvv, Zovoo, Zvvvo, omega)

                r1 += z1; r2 += z2

            self.X1 += r1/(Dia + omega)
            self.X2 += r2/(Dijab + omega)

            rms = contract('ia,ia->', r1/(Dia+omega), r1/(Dia+omega))
            rms += contract('ijab,ijab->', r2/(Dijab+omega), r2/(Dijab+omega))
            rms = np.sqrt(rms)

            pseudo = self.pseudoresponse(pertbar, self.X1, self.X2)
            pseudodiff = pseudo - pseudo_last
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds." % (time.time() - solver_start))
                print("\nLargest X amplitudes:")
                print_wfn(self.X1, self.X2)

                if self.ccwfn.model == 'CC3' and store_triples is True:
                    return [omega, self.X1, self.X2, self.X3], pseudo
                else:
                    return [omega, self.X1, self.X2], pseudo

            diis.add_error_vector(self.X1, self.X2)
            if niter >= start_diis:
                self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)

    def r_X1(self, pertbar, omega):
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        hbar = self.hbar

        r_X1 = (pertbar.Avo.T - omega * X1).copy()
        r_X1 += contract('ie,ae->ia', X1, hbar.Hvv)
        r_X1 -= contract('ma,mi->ia', X1, hbar.Hoo)
        r_X1 += contract('me,maei->ia', X1, hbar.Hovvo)
        r_X1 += contract('me,imae->ia', hbar.Hov, X2)
        r_X1 += (1/2) * contract('imef,amef->ia', X2, hbar.Hvovv)
        r_X1 -= (1/2) * contract('mnae,mnie->ia', X2, hbar.Hooov)

        return r_X1

    def r_X2(self, pertbar, omega):
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        t2 = self.ccwfn.t2
        hbar = self.hbar
        ERI = self.H.ERI

        # Intermediates to handle three-body HBAR contributions
        # TODO: combine to reduce number of contractions
        Zvv = contract('amef,me->af', hbar.Hvovv, X1)
        Zoo = contract('mnie,me->ni', hbar.Hooov, X1)
        Yoo = (1/2) * contract('mnef,mjef->nj', ERI[o,o,v,v], X2)
        Yvv = (1/2) * contract('mnef,mneb->fb', ERI[o,o,v,v], X2)

        r_X2 = (pertbar.Avvoo - omega*X2).copy()
        r_X2 += contract('ie,abej->ijab', X1, hbar.Hvvvo) - contract('je,abei->ijab', X1, hbar.Hvvvo)
        r_X2 -= contract('ma,mbij->ijab', X1, hbar.Hovoo) - contract('mb,maij->ijab', X1, hbar.Hovoo)
        r_X2 += contract('ni,njab->ijab', Zoo, t2) - contract('nj,niab->ijab', Zoo, t2)
        r_X2 -= contract('af,ijfb->ijab', Zvv, t2) - contract('bf,ijfa->ijab', Zvv, t2)
        r_X2 -= contract('nj,inab->ijab', Yoo, t2) - contract('ni,jnab->ijab', Yoo, t2)
        r_X2 -= contract('fb,ijaf->ijab', Yvv, t2) - contract('fa,ijbf->ijab', Yvv, t2)
        r_X2 += contract('ijae,be->ijab', X2, hbar.Hvv) - contract('ijbe,ae->ijab', X2, hbar.Hvv)
        r_X2 -= contract('imab,mj->ijab', X2, hbar.Hoo) - contract('jmab,mi->ijab', X2, hbar.Hoo)
        r_X2 += (1/2) * contract('mnab,mnij->ijab', X2, hbar.Hoooo)
        r_X2 += (1/2) * contract('ijef,abef->ijab', X2, hbar.Hvvvv)
        tmp = contract('imae,mbej->ijab', X2, hbar.Hovvo)
        r_X2 += tmp - tmp.swapaxes(0,1) - tmp.swapaxes(2,3) + tmp.swapaxes(0,1).swapaxes(2,3)

        return r_X2


    def pseudoresponse(self, pertbar, X1, X2):
        polar1 = contract('ai,ia->', pertbar.Avo, X1)
        polar2 = (1/4) * contract('ijab,ijab->', pertbar.Avvoo, X2)
        return -2.0*(polar1 + polar2)


    # CC3 intermediates that may be pre-computed:
    # (1) <mu2|[[H^,T3],X1]|0> --> X2
    #     We compute Yoovo and Yovvv that do not depend on X1 to avoid iterating T3 unnecessarily.
    #     [Cf. the CC3_noniter() function in cclambda.py, but note that antisymmetrization is required here for the 
    #     contraction with X1.]
    # (2) <mu3|[[A,T2],T2]|0> --> X3
    #     We compute the contraction of A against one of the T2s to generate Zovoo and Zvvvo that can be dropped 
    #     into the connected T3-driver.
    def CC3_noniter(self, o, v, t2, F, ERI, Wvvvo, Wovoo, pertbar):
        Yoovo = np.zeros_like(ERI[o,o,v,o])
        Yovvv = np.zeros_like(ERI[o,v,v,v])
        no = t2.shape[0]
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = t3c_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)
                    Yoovo[i,j] -= (1/2) * contract('abc,lbc->al', t3, ERI[o,k,v,v])
                    Yovvv[i] -= (1/2) * contract('abc,dc->abd', t3, ERI[j,k,v,v])

        Zovoo = (1/2) * contract('ld,jkdc->lcjk', pertbar.Aov, t2)
        Zvvvo = -(1/2) * contract('ld,lkbc->bcdk', pertbar.Aov, t2)

        return Yoovo, Yovvv, Zovoo, Zvvvo

    # CC3 intermediates and T3/X3 contributions to X1 and X2 equations
    # These must be computed in every iteration for building the perturbed wave functions
    # [cf the CC3_noniter() function in cclambda.py]
    def CC3_iter(self, o, v, X1, X2, t2, F, ERI, hbar, pert, Fov, Wvvvv, Woooo, Wovvo, Wvvvo, Wovoo, Yoovo, Yovvv, Zovoo, Zvvvo, omega):
        no = X1.shape[0]
        nv = X1.shape[1]

        z1 = np.zeros_like(X1)
        z2 = np.zeros_like(X2)

        # <mu2|[[H^,T3],X1]|0> --> X2
        Yov = contract('ld,klcd->kc', X1, ERI[o,o,v,v])
        tmp = contract('ld,ijal->ijad', X1, Yoovo)
        z2 += tmp - tmp.swapaxes(2,3)
        tmp = contract('ld,iabd->ilab', X1, Yovvv)
        z2 += tmp - tmp.swapaxes(0,1)

        # <mu3|[[H^,T2,X1]|0> --> X3
        Zbcdk = contract('ke,bcde->bcdk', X1, Wvvvv)
        tmp = -contract('lb,lcdk->bcdk', X1, Wovvo)
        Zbcdk += tmp - tmp.swapaxes(0,1)
        Zlcjk = -contract('mc,lmjk->lcjk', X1, Woooo)
        tmp = contract('jd,lcdk->lcjk', X1, Wovvo)
        Zlcjk += tmp - tmp.swapaxes(2,3)

        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    # <mu2|[[H^,T3],X1]|0> --> X2 (remaining term)
                    t3 = t3c_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)
                    z2[i,j] += contract('abc,c->ab', t3, Yov[k])

                    # <mu3|[ABAR,T3]|0> --> X3
                    tmp = contract('abc,dc->abd', t3, pert.Avv)
                    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
                    denom = np.zeros_like(t3)
                    denom -= vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir
                    denom += occ[i] + occ[j] + occ[k]
                    denom += omega
                    x3 = x3/denom

                    # <mu3|[[ABAR,T2],T2]|0> + <mu3|[[H^,T2,X1]|0> --> X3
                    x3 += t3c_ijk(o, v, i, j, k, t2, F, Zvvvo+Zbcdk, Zovoo+Zlcjk, omega)

                    # <mu3|[H^,X2]|0> --> X3
                    x3 += t3c_ijk(o, v, i, j, k, X2, F, Wvvvo, Wovoo, omega)

                    z1[i] += (1/4) * contract('abc,bc->a', x3, ERI[j,k,v,v])
                    z2[i,j] += contract('abc,c->ab', x3, hbar.Hov[k])
                    tmp = (1/2) * contract('abc,dbc->ad', x3, hbar.Hvovv[:,k,:,:])
                    z2[i,j] += tmp - tmp.swapaxes(0,1)
                    for l in range(no):
                        tmp = -(1/2) * contract('abc,c->ab', x3, hbar.Hooov[j,k,l,:])
                        z2[i,l] += tmp
                        z2[l,i] -= tmp

        y1 = np.zeros_like(z1.T)
        y2 = np.zeros_like(z2.T)
        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    # <mu3|[ABAR,T3]|0> --> X3
                    t3 = t3c_abc(o, v, a, b, c, t2, F, Wvvvo, Wovoo)
                    tmp = -contract('ijk,kl->ijl', t3, pert.Aoo)
                    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
                    denom = np.zeros_like(t3)
                    denom += occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ
                    denom -= vir[a] + vir[b] + vir[c]
                    denom += omega
                    x3 = x3/denom

                    y1[a] += (1/4) * contract('ijk,jk->i', x3, ERI[o,o,b+no,c+no])
                    y2[a,b] += contract('ijk,k->ij', x3, hbar.Hov[:,c])
                    tmp = -(1/2) * contract('ijk,jkl->il', x3, hbar.Hooov[:,:,:,c])
                    y2[a,b] += tmp - tmp.swapaxes(0,1)
                    for d in range(nv):
                        tmp = (1/2) * contract('ijk,k->ij', x3, hbar.Hvovv[d,:,b,c])
                        y2[a,d] += tmp
                        y2[d,a] -= tmp

        z1 += y1.T
        z2 += y2.T

        return z1, z2

    def CC3_iter_full(self, o, v, X1, X2, t2, F, ERI, hbar, pertbar, Wvvvv, Woooo, Wovvo, Wvvvo, Wovoo, omega):
        no = X1.shape[0]
        nv = X1.shape[1]

        t3 = self.ccwfn.t3

        # <mu3|[ABAR,T3]|0>
        tmp = contract('ijkabc,dc->ijkabd', t3, pertbar.Avv)
        X3 = tmp - tmp.swapaxes(3,5) - tmp.swapaxes(4,5)
        tmp = -contract('ijkabc,kl->ijlabc', t3, pertbar.Aoo)
        X3 += tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)

        # 1/2 <mu3|[[A,T2],T2]|0>
        tmp = contract('lkbc,ld->bcdk', t2, pertbar.Aov)
        X3_a = -contract('ijad,bcdk->ijkabc', t2, tmp)
        # <mu3|[[H^,T2],X1]|0>
        Zbcdk = contract('ke,bcde->bcdk', X1, Wvvvv)
        tmp = -contract('lb,lcdk->bcdk', X1, Wovvo)
        Zbcdk += tmp - tmp.swapaxes(0,1)
        X3_a += contract('ijad,bcdk->ijkabc', t2, Zbcdk)
        # <mu3|[H^,X2]|0>
        X3_a += contract('ijad,bcdk->ijkabc', X2, Wvvvo)

        # P(k/ij) P(a/bc) f(ijkabc)
        X3 += permute_triples(X3_a, 'k/ij', 'a/bc')

        # <mu3|[[H^,T2],X1]|0>
        Zlcjk = contract('mc,lmjk->lcjk', X1, Woooo)
        tmp = -contract('jd,lcdk->lcjk', X1, Wovvo)
        Zlcjk += tmp - tmp.swapaxes(2,3)
        X3_b = contract('ilab,lcjk->ijkabc', t2, Zlcjk)
        # <mu3|[H^,X2]|0>
        X3_b += -contract('ilab,lcjk->ijkabc', X2, Wovoo)

        # P(i/jk) P(c/ab) f(ijkabc)
        X3 += permute_triples(X3_b, 'i/jk', 'c/ab')

        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        denom = occ.reshape(-1,1,1,1,1,1) + occ.reshape(-1,1,1,1,1) + occ.reshape(-1,1,1,1) - vir.reshape(-1,1,1) - vir.reshape(-1,1) - vir
        denom += omega
        X3 = X3/denom

        # Save the X3s so they can be stored in the perturbation dictionary for computing the response functions
        self.X3 = X3

        # <mu1|[H,X3]|0>
        z1 = (1/4) * contract('ijkabc,jkbc->ia', X3, ERI[o,o,v,v])

        # <mu2|[[H,T3],X1]|0>
        tmp = contract('ld,klcd->kc', X1, ERI[o,o,v,v])
        z2 = contract('ijkabc,kc->ijab', t3, tmp)

        tmp = contract('ld,jlbc->djbc', X1, ERI[o,o,v,v])
        tmp = -(1/2) * contract('ijkabc,djbc->ikad', t3, tmp)
        z2 += tmp - tmp.swapaxes(2,3)

        tmp = contract('ld,jkbd->jklb', X1, ERI[o,o,v,v])
        tmp = -(1/2) * contract('ijkabc,jklb->ilac', t3, tmp)
        z2 += tmp - tmp.swapaxes(0,1)

        # <mu2|[HBAR,X3]|0>
        z2 += contract('ijkabc,kc->ijab', X3, hbar.Hov)
        tmp = (1/2) * contract('ijkabc,dkbc->ijad', X3, hbar.Hvovv)
        z2 += tmp - tmp.swapaxes(2,3)
        tmp = -(1/2) * contract('ijkabc,jklc->ilab', X3, hbar.Hooov)
        z2 += tmp - tmp.swapaxes(0,1)

        return z1, z2


    # Compute CC3-related intermediates needed for build X3 for the linear response function
    # This function is used by the linear response code, even though these intermediates 
    # are also needed for computing the perturbed X3 amplitudes.
    def CC3_linresp_intermediates(self, o, v, pert, t1, t2, X1, ERI):

        Woooo = self.ccwfn.build_Woooo_CC3(o, v, ERI, t1)
        Wvvvv = self.cclambda.build_Wvvvv_CC3(o, v, ERI, t1)
        Wovvo = self.cclambda.build_Wovvo_CC3(o, v, ERI, t1)

        Zvvvo = -(1/2) * contract('ld,lkbc->bcdk', pert.Aov, t2)
        Zvvvo += contract('ke,bcde->bcdk', X1, Wvvvv)
        tmp = -contract('lb,lcdk->bcdk', X1, Wovvo)
        Zvvvo += tmp - tmp.swapaxes(0,1)

        Zovoo = (1/2) * contract('ld,jkdc->lcjk', pert.Aov, t2)
        Zovoo -= contract('mc,lmjk->lcjk', X1, Woooo)
        tmp = contract('jd,lcdk->lcjk', X1, Wovvo)
        Zovoo += tmp - tmp.swapaxes(2,3)

        return Zvvvo, Zovoo


    def pertcheck(self, omega, e_conv=1e-13, r_conv=1e-13, maxiter=200, max_diis=8, start_diis=1):
        """
        Build first-order perturbed wave functions for all available perturbations and return a dict of their converged pseudoresponse values.  Primarily for testing purposes.

        Parameters
        ----------
        omega: float
            The external field frequency.
        e_conv : float
            convergence condition for the pseudoresponse value (default if 1e-13)
        r_conv : float
            convergence condition for perturbed wave function rmsd (default if 1e-13)
        maxiter : int
            maximum allowed number of iterations of the wave function equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        check: dictionary
            Converged pseudoresponse values for all available perturbations.
        """

        # Build dictionary of all similarity-transformed property integrals for testing
        self.pertbar = {}

        # Electric-dipole operator (length)
        for axis in range(1,2):
            key = "MU_" + self.cpolar_LCX_CC3art[axis]
            self.pertbar[key] = pertbar(self.H.mu[axis], self.ccwfn)

#        # Magnetic-dipole operator
#        for axis in range(3):
#            key = "M_" + self.cart[axis]
#            self.pertbar[key] = pertbar(self.H.m[axis], self.ccwfn)
#
#        # Complex-conjugate of magnetic-dipole operator
#        for axis in range(3):
#            key = "M*_" + self.cart[axis]
#            self.pertbar[key] = pertbar(-1.0*self.H.m[axis], self.ccwfn)
#
#        # Electric-dipole operator (velocity)
#        for axis in range(3):
#            key = "P_" + self.cart[axis]
#            self.pertbar[key] = pertbar(self.H.p[axis], self.ccwfn)
#
#        # Complex-conjugate of electric-dipole operator (velocity)
#        for axis in range(3):
#            key = "P*_" + self.cart[axis]
#            self.pertbar[key] = pertbar(-1.0*self.H.p[axis], self.ccwfn)
#
#        # Traceless quadrupole
#        ij = 0
#        for axis1 in range(3):
#            for axis2 in range(axis1,3):
#                key = "Q_" + self.cart[axis1] + self.cart[axis2]
#                self.pertbar[key] = pertbar(self.H.Q[ij], self.ccwfn)
#                if (axis1 != axis2):
#                    key2 = "Q_" + self.cart[axis2] + self.cart[axis1]
#                    self.pertbar[key2] = self.pertbar[key]
#                ij += 1

        # dictionaries for perturbed wave functions and test pseudoresponses
        X = {}
        check = {}

        # Electric-dipole (length)
        for axis in range(1,2):
            pertkey = "MU_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
#            if (omega != 0.0):
#                X_key = pertkey + "_" + f"{-omega:0.6f}"
#                print("Solving right-hand perturbed wave function for %s:" % (X_key))
#                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#                check[X_key] = polar
#
#        # Magnetic-dipole
#        for axis in range(3):
#            pertkey = "M_" + self.cart[axis]
#            X_key = pertkey + "_" + f"{omega:0.6f}"
#            print("Solving right-hand perturbed wave function for %s:" % (X_key))
#            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#            check[X_key] = polar
#            if (omega != 0.0):
#                X_key = pertkey + "_" + f"{-omega:0.6f}"
#                print("Solving right-hand perturbed wave function for %s:" % (X_key))
#                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#                check[X_key] = polar
#
#        # Complex-conjugate of magnetic-dipole
#        for axis in range(3):
#            pertkey = "M*_" + self.cart[axis]
#            X_key = pertkey + "_" + f"{omega:0.6f}"
#            print("Solving right-hand perturbed wave function for %s:" % (X_key))
#            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#            check[X_key] = polar
#            if (omega != 0.0):
#                X_key = pertkey + "_" + f"{-omega:0.6f}"
#                print("Solving right-hand perturbed wave function for %s:" % (X_key))
#                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#                check[X_key] = polar
#
#        # Electric-dipole (velocity)
#        for axis in range(3):
#            pertkey = "P_" + self.cart[axis]
#            X_key = pertkey + "_" + f"{omega:0.6f}"
#            print("Solving right-hand perturbed wave function for %s:" % (X_key))
#            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#            check[X_key] = polar
#            if (omega != 0.0):
#                X_key = pertkey + "_" + f"{-omega:0.6f}"
#                print("Solving right-hand perturbed wave function for %s:" % (X_key))
#                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#                check[X_key] = polar
#
#        # Complex-conjugate of electric-dipole (velocity)
#        for axis in range(3):
#            pertkey = "P*_" + self.cart[axis]
#            X_key = pertkey + "_" + f"{omega:0.6f}"
#            print("Solving right-hand perturbed wave function for %s:" % (X_key))
#            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#            check[X_key] = polar
#            if (omega != 0.0):
#                X_key = pertkey + "_" + f"{-omega:0.6f}"
#                print("Solving right-hand perturbed wave function for %s:" % (X_key))
#                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#                check[X_key] = polar
#
#        # Traceless quadrupole
#        for axis1 in range(3):
#            for axis2 in range(3):
#                pertkey = "Q_" + self.cart[axis1] + self.cart[axis2]
#                X_key = pertkey + "_" + f"{omega:0.6f}"
#                print("Solving right-hand perturbed wave function for %s:" % (X_key))
#                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#                check[X_key] = polar
#                if (omega != 0.0):
#                    X_key = pertkey + "_" + f"{-omega:0.6f}"
#                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
#                    X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
#                    check[X_key] = polar

        return check


class pertbar(object):
    """
    A class for building and storing similarity-transformed one-electron property operators.
    """

    def __init__(self, pert, ccwfn):
        """
        Parameters:
        -----------
        pert: one-electron property integrals (see constructor for ccresponse object)
        ccwfn: object
        """
        o = ccwfn.o
        v = ccwfn.v
        t1 = ccwfn.t1
        t2 = ccwfn.t2

        self.Aov = pert[o,v].copy()

        self.Aoo = pert[o,o].copy()
        self.Aoo += contract('ie,me->mi', t1, pert[o,v])

        self.Avv = pert[v,v].copy()
        self.Avv -= contract('ma,me->ae', t1, pert[o,v])

        self.Avo = pert[v,o].copy()
        self.Avo += contract('ie,ae->ai', t1, pert[v,v])
        self.Avo -= contract('ma,mi->ai', t1, pert[o,o])
        self.Avo += contract('miea,me->ai', t2, pert[o,v])
        self.Avo -= contract('ie,ma,me->ai', t1, t1, pert[o,v])

        self.Aovoo = contract('ijeb,me->mbij', t2, pert[o,v])

        self.Avvvo = -1.0 * contract('miab,me->abei', t2, pert[o,v])

        # Note that Avvoo is stored oovv in order to match X2/t2/l2 ordering
        self.Avvoo = contract('ijae,be->ijab', t2, self.Avv) - contract('ijbe,ae->ijab', t2, self.Avv)
        self.Avvoo -= contract('imab,mj->ijab', t2, self.Aoo) - contract('jmab,mi->ijab', t2, self.Aoo)

        if ccwfn.model == 'CC3':
            if ccwfn.store_triples is True:
                self.Avvoo += contract('ijkabc,kc->ijab', ccwfn.t3, pert[o,v])
            else:
                F = ccwfn.H.F
                ERI = ccwfn.H.ERI
                Woooo = ccwfn.build_Woooo_CC3(o, v, ERI, t1)
                Wovoo = ccwfn.build_Wovoo_CC3(o, v, ERI, t1, Woooo)
                Wvvvo = ccwfn.build_Wvvvo_CC3(o, v, ERI, t1)

                x2 = np.zeros_like(t2)
                no = ccwfn.no
                for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            t3 = t3c_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)
                            self.Avvoo[i,j] += contract('c,abc->ab', pert[k,v], t3)

