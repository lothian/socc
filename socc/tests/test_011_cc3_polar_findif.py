import psi4
import socc
import pytest
import numpy as np
from ..data.molecules import *

def test_cc3_polar_findif():
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'STO-3G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'reference' : 'rhf',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12,
                      'omega': [0.0],
                      'diis': 1})

    F = 0.0001
    axis = 'Z'
    e_conv = 1e-12
    r_conv = 1e-12
    maxiter = 75

    model = 'CC3'
    mol = psi4.geometry(moldict['H2O'])

    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model=model)
    e0 = cc_wfn.solve_cc(e_conv,r_conv,maxiter, store_triples=True)

    pert = F
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    mints = psi4.core.MintsHelper(scf_wfn)
    cc_wfn = socc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    ep = cc_wfn.solve_cc(e_conv,r_conv,maxiter, store_triples=True)

    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    pert = 2*F
    cc_wfn = socc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    e2p = cc_wfn.solve_cc(e_conv,r_conv,maxiter, store_triples=True)

    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    pert = -F
    cc_wfn = socc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    em = cc_wfn.solve_cc(e_conv,r_conv,maxiter, store_triples=True)

    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    pert = -2*F
    cc_wfn = socc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    e2m = cc_wfn.solve_cc(e_conv,r_conv,maxiter, store_triples=True)

    print("E(0)   = %20.15f" % e0)
    print("E(+F)  = %20.15f" % ep)
    print("E(+2F) = %20.15f" % e2p)
    print("E(-F)  = %20.15f" % em)
    print("E(-2F) = %20.15f" % e2m)

    # Compute dipole moment
    mu_z = -(-e2p + 8*ep - 8*em + e2m)/(12*F)

    # compute polarizability
    alpha_zz = -(-e2p + 16*ep - 30*e0 + 16*em - e2m)/(12*F*F)
    print(alpha_zz)
    print("Dalton alpha_zz = 2.9989468.")
