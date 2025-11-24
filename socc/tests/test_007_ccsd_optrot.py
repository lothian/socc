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
    epsi4 = -0.097061805835190
    lpsi4 = -0.094659751195269
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)
    ccresp = socc.ccresponse(cc_wfn, lambda_wfn)
    omega = 0.077357 # 589 nm
    optrot = ccresp.optrot(omega)

    #psi4.properties('CCSD', properties=['rotation'])

    psi4_optrot = np.array([[-0.012969077546209,  0.238413335857954, 0.000000000000000],
                            [-0.039665773503841, 0.124406421486684, -0.000000000000000],
                            [-0.000000000000000, 0.000000000000000, -0.112853202255656]])

    assert(np.allclose(psi4_optrot, optrot, 1e-10))

    # cc-pVDZ basis set
    psi4.set_options({'basis': 'cc-pVDZ'})
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CCSD')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    hbar = socc.cchbar(cc_wfn)
    lambda_wfn = socc.cclambda(cc_wfn, hbar)
    lcc = lambda_wfn.solve_lambda(e_conv, r_conv)
    epsi4 = -0.394679216778747
    lpsi4 = -0.386314800298656
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)
    ccresp = socc.ccresponse(cc_wfn, lambda_wfn)
    omega = 0.077357  # 589 nm
    optrot = ccresp.optrot(omega)

    #psi4.properties('CCSD', properties=['rotation'])

    psi4_optrot = np.array([[-0.012765691587814, 0.266267947958225,  0.000000000000002],
                            [ 0.064272290616867, 0.155734081606032, -0.000000000000026],
                            [-0.000000000000004, 0.000000000000018, -0.127276678468221]])

    assert(np.allclose(psi4_optrot, optrot, 1e-10))
