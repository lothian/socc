if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import numpy as np


class hamiltonian(object):
    """
    A molecular spin-orbital Hamiltonian object.

    Attributes
    ----------
    F : NumPy array
        Spin-orbital-basis Fock matrix (can be non-diagonal)
    ERI : NumPy array
        Anti-symmetrized spin-orbital-basis electron repulsion integrals in Dirac notation: <pq||rs>
    mu : NumPy array
        Spin-orbital-basis electric dipole integrals (length)
    m : NumPy array
        Spin-orbital-basis magnetic dipole integrals
    """
    def __init__(self, ref, Ca, Cb, spin, spat):

        npCa = np.asarray(Ca)
        npCb = np.asarray(Cb)
        nact = spat.shape[0]

        # Spin-orbital Fock matrix
        Fa = np.asarray(ref.Fa())
        Fa = npCa.T @ Fa @ npCa
        Fb = np.asarray(ref.Fb())
        Fb = npCb.T @ Fb @ npCb

        F = np.zeros((nact,nact))
        for p in range(nact):
            pspin = spin[p]
            pspat = spat[p]
            for q in range(nact):
                qspin = spin[q]
                qspat = spat[q]
                if pspin==0 and qspin==0:
                    F[p,q] = Fa[pspat,qspat]
                elif pspin == 1 and qspin==1:
                    F[p,q] = Fb[pspat,qspat]

        self.F = F

        # Spin-orbital two-electron integrals in antisymmetrized Dirac notation
        mints = psi4.core.MintsHelper(ref.basisset())
        ERI_AA = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca))  # (pr|qs)
        ERI_BB = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb))  # (pr|qs)
        ERI_AB = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))  # (pr|qs)

        ERI = np.zeros((nact, nact, nact, nact))
        for p in range(nact):
            pspin = spin[p]
            pspat = spat[p]
            for q in range(nact):
                qspin = spin[q]
                qspat = spat[q]
                for r in range(nact):
                    rspin = spin[r]
                    rspat = spat[r]
                    for s in range(nact):
                        sspin = spin[s]
                        sspat = spat[s]

                        if pspin==0 and qspin==0 and rspin==0 and sspin==0:
                            ERI[p,q,r,s] = ERI_AA[pspat, rspat, qspat, sspat] - ERI_AA[pspat, sspat, qspat, rspat]
                        elif pspin==1 and qspin==1 and rspin==1 and sspin==1:
                            ERI[p,q,r,s] = ERI_BB[pspat, rspat, qspat, sspat] - ERI_BB[pspat, sspat, qspat, rspat]
                        elif pspin==0 and qspin==1 and rspin==0 and sspin==1:
                            ERI[p,q,r,s] = ERI_AB[pspat, rspat, qspat, sspat]
                        elif pspin==1 and qspin==0 and rspin==1 and sspin==0:
                            ERI[p,q,r,s] = ERI_AB[qspat, sspat, pspat, rspat]
                        elif pspin==0 and qspin==1 and rspin==1 and sspin==0:
                            ERI[p,q,r,s] = -ERI_AB[pspat, sspat, qspat, rspat]
                        elif pspin==1 and qspin==0 and rspin==0 and sspin==1:
                            ERI[p,q,r,s] = -ERI_AB[qspat, rspat, pspat, sspat]
        self.ERI = ERI

        ## One-electron property integrals

        # Electric dipole integrals (length): -e r
        dipole_ints = mints.ao_dipole()
        self.mu = []
        for axis in range(3):
            mu_a = npCa.T @ np.asarray(dipole_ints[axis]) @ npCa
            mu_b = npCb.T @ np.asarray(dipole_ints[axis]) @ npCb
            mu = np.zeros((nact,nact))
            for p in range(nact):
                pspin = spin[p]
                pspat = spat[p]
                for q in range(nact):
                    qspin = spin[q]
                    qspat = spat[q]
                    if pspin==0 and qspin==0:
                        mu[p,q] = mu_a[pspat,qspat]
                    elif pspin == 1 and qspin==1:
                        mu[p,q] = mu_b[pspat,qspat]
            self.mu.append(mu)

        # Magnetic dipole integrals: -(e/2 m_e) L (pure imaginary, but stored as real)
        m_ints = mints.ao_angular_momentum()
        self.m = []
        for axis in range(3):
            m_a = npCa.T @ (np.asarray(m_ints[axis])*-0.5) @ npCa
            m_b = npCb.T @ (np.asarray(m_ints[axis])*-0.5) @ npCb
            m = np.zeros((nact,nact))
            for p in range(nact):
                pspin = spin[p]
                pspat = spat[p]
                for q in range(nact):
                    qspin = spin[q]
                    qspat = spat[q]
                    if pspin==0 and qspin==0:
                        m[p,q] = m_a[pspat,qspat]
                    elif pspin == 1 and qspin==1:
                        m[p,q] = m_b[pspat,qspat]
            self.m.append(m)

        # Linear momentum integrals: (-e) (-i hbar) Del (pure imaginary, but stored as real)
        p_ints = mints.ao_nabla()
        self.p = []
        for axis in range(3):
            p_a = npCa.T @ np.asarray(p_ints[axis]) @ npCa
            p_b = npCb.T @ np.asarray(p_ints[axis]) @ npCb
            p_spin = np.zeros((nact,nact))
            for p in range(nact):
                pspin = spin[p]
                pspat = spat[p]
                for q in range(nact):
                    qspin = spin[q]
                    qspat = spat[q]
                    if pspin==0 and qspin==0:
                        p_spin[p,q] = p_a[pspat,qspat]
                    elif pspin == 1 and qspin==1:
                        p_spin[p,q] = p_b[pspat,qspat]
            self.p.append(p_spin)

        # Traceless quadrupole: 
        Q_ints = mints.ao_traceless_quadrupole()
        self.Q = []
        ij = 0
        for axis1 in range(3):
            for axis2 in range(axis1,3):
                Q_a = npCa.T @ np.asarray(Q_ints[ij]) @ npCa
                Q_b = npCb.T @ np.asarray(Q_ints[ij]) @ npCb
                Q = np.zeros((nact,nact))
                for p in range(nact):
                    pspin = spin[p]
                    pspat = spat[p]
                    for q in range(nact):
                        qspin = spin[q]
                        qspat = spat[q]
                        if pspin==0 and qspin==0:
                            Q[p,q] = Q_a[pspat,qspat]
                        elif pspin == 1 and qspin==1:
                            Q[p,q] = Q_b[pspat,qspat]
                self.Q.append(Q)
                ij += 1
