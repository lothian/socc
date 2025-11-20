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

    model = 'CCSD'
    mol = psi4.geometry(moldict['H2O'])
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)

    cc_wfn = pycc.ccwfn(scf_wfn, model=model)
    e0 = ccsd.solve_cc(e_conv,r_conv,maxiter)

    pert = F
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    mints = psi4.core.MintsHelper(scf_wfn)
    ccsd = pycc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    ep = ccsd.solve_cc(e_conv,r_conv,maxiter)

    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    pert = 2*F
    ccsd = pycc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    e2p = ccsd.solve_cc(e_conv,r_conv,maxiter)

    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    pert = -F
    ccsd = pycc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    em = ccsd.solve_cc(e_conv,r_conv,maxiter)

    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    pert = -2*F
    ccsd = pycc.ccwfn(scf_wfn, model=model, field=True, field_strength=pert, field_axis=axis)
    e2m = ccsd.solve_cc(e_conv,r_conv,maxiter)


# Compute dipole moment
mu_z = -(ep - em)/(2*pert)
print(mu_z)

# compute polarizability
alpha_zz = -(-e2p + 16*ep - 30*e0 + 16*em - e2m)/(12*F*F)
print(alpha_zz)

#psi4.properties('CCSD', properties=['polarizability'])
