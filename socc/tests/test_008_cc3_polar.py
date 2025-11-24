import psi4
import socc
import pytest
import numpy as np
from ..data.molecules import *

def test_cc3_polar():
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
                    'omega' : [0.1],
                    'diis': 1})

    e_conv = 1e-12
    r_conv = 1e-12

    mol = psi4.geometry(moldict["H2O"])
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv, store_triples=True)
    hbar = socc.cchbar(cc_wfn)
    lambda_wfn = socc.cclambda(cc_wfn, hbar)
    lcc = lambda_wfn.solve_lambda(e_conv, r_conv)
    epsi4 = -0.070778085758433
    lpsi4 = -0.068979529552146
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)
    ccresp = socc.ccresponse(cc_wfn, lambda_wfn)
    omega = 0.1
    polar = ccresp.polarizability(omega)

    dalton_polar = np.array([[ 0.061593757,  0.0000000, 0.0000000],
                             [ 0.000000000,  7.0661684, 0.0000000],
                             [ 0.000000000,  0.0000000, 3.0604929]])
    assert(np.allclose(dalton_polar, polar, 1e-6, 1e-6))
