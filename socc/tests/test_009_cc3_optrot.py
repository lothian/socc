import psi4
import socc
import pytest
import numpy as np
from ..data.molecules import *

def test_cc3_optrot():
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
                    'omega' : [0.077357],
                    'gauge' : 'length',
                    'diis': 1})

    e_conv = 1e-12
    r_conv = 1e-12

    mol = psi4.geometry(moldict["H2O2"])
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv, store_triples=True)
    hbar = socc.cchbar(cc_wfn)
    lambda_wfn = socc.cclambda(cc_wfn, hbar)
    lcc = lambda_wfn.solve_lambda(e_conv, r_conv)
    epsi4 = -0.097663033465501
    lpsi4 = -0.095210691156924
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)
    ccresp = socc.ccresponse(cc_wfn, lambda_wfn)
    omega = 0.077357 # 589 nm
    optrot = ccresp.optrot(omega)

    dalton_optrot = np.array([[ 0.124157,  0.000000,  0.000000],
                              [ 0.000000, -0.012797,  0.000000],
                              [ 0.000000,  0.000000, -0.113186]])
    assert(np.allclose(np.diag(dalton_optrot), np.diag(optrot), 1e-4, 1e-5))
