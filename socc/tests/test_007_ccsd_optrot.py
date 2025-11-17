import psi4
import socc
import pytest
import numpy as np
from ..data.molecules import *

def test_ccsd_optrot():
    psi4.core.clean()
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'STO-3G',
                    'scf_type': 'pk',
                    'mp2_type': 'conv',
                    'freeze_core': 'true',
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
    cc_wfn = socc.ccwfn(scf_wfn, model='CCSD')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    hbar = socc.cchbar(cc_wfn)
    lambda_wfn = socc.cclambda(cc_wfn, hbar)
    lcc = lambda_wfn.solve_lambda(e_conv, r_conv)
    epsi4 = -0.097061805905703
    lpsi4 = -0.094659751261054
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)
    ccresp = socc.ccresponse(cc_wfn, lambda_wfn)
    omega = 0.077357 # 589 nm
    optrot = ccresp.optrot(omega)

#    psi4.properties('CCSD', properties=['rotation'])

    psi4_optrot = np.array([[ 0.124406421539390,  0.039665773574171, -0.000000000000002],
                            [-0.238413336061875, -0.012969077558462,  0.000000000000001],
                            [ 0.000000000000004, -0.000000000000001, -0.112853202325508]])
    assert(np.allclose(psi4_optrot, optrot, 1e-10))

    # cc-pVDZ basis set
    psi4.set_options({'basis': 'cc-pVDZ'})
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CCSD')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    hbar = socc.cchbar(cc_wfn)
    lambda_wfn = socc.cclambda(cc_wfn, hbar)
    lcc = lambda_wfn.solve_lambda(e_conv, r_conv)
    epsi4 = -0.394679216819712
    lpsi4 = -0.386314800333307
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)
    ccresp = socc.ccresponse(cc_wfn, lambda_wfn)
    omega = 0.077357  # 589 nm
    optrot = ccresp.optrot(omega)

#    psi4.properties('CCSD', properties=['rotation'])

    psi4_optrot = np.array([[ 0.155734081834151, -0.064272290708449, -0.000000000000058],
                            [-0.266267948269857, -0.012765691621003,  0.000000000000006],
                            [ 0.000000000000033,  0.000000000000007, -0.127276678710314]])
    assert(np.allclose(psi4_optrot, optrot, 1e-10))
