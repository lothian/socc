import psi4
import socc
import pytest
from ..data.molecules import *

def test_cc3_lambda():
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

    epsi4 = -0.0707148636207360
    lpsi4 = -0.0689177907842559

    mol = psi4.geometry(moldict["H2O"])
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv, store_triples=True)
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv,  alg='AB')
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv,  alg='IJ')
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    # cc-pVDZ basis set
    epsi4 = -0.225982879345687
    lpsi4 = -0.221452780028464

    psi4.core.clean()
    psi4.set_options({'basis': 'cc-pVDZ'})
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv, store_triples=True)
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv,  alg='AB')
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv,  alg='IJ')
    hbar = socc.cchbar(cc_wfn)
    cclambda = socc.cclambda(cc_wfn, hbar)
    lcc = cclambda.solve_lambda(e_conv, r_conv)
    assert (abs(epsi4 - ecc) < 1e-11)
    assert (abs(lpsi4 - lcc) < 1e-11)

