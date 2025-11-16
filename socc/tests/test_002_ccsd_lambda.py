import psi4
import socc
import pytest
from ..data.molecules import *

def test_ccsd_h2o():
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
                    'diis': 1})

    e_conv = 1e-12
    r_conv = 1e-12

    mol = psi4.geometry(moldict["H2O_Pycc"])
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CCSD')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    epsi4 = -0.070616830152761
    lpsi4 = -0.068826452648939
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    # cc-pVDZ basis set
    psi4.set_options({'basis': 'cc-pVDZ'})
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CCSD')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    epsi4 = -0.222029814166783
    lpsi4 = -0.217838951550509
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)
