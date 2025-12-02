import psi4
import socc
import pytest
from ..data.molecules import *

def test_cc3_energy():
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
    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    epsi4 = -0.070714823977789
    assert (abs(epsi4 - ecc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv, store_triples=True)
    assert (abs(epsi4 - ecc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv, alg='AB')
    assert (abs(epsi4 - ecc) < 1e-11)

    # cc-pVDZ basis set
    psi4.core.clean()
    psi4.set_options({'basis': 'cc-pVDZ'})
    scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv)
    epsi4 = -0.225982859180558
    assert (abs(epsi4 - ecc) < 1e-11)

    cc_wfn = socc.ccwfn(scf_wfn, model='CC3')
    ecc = cc_wfn.solve_cc(e_conv, r_conv, store_triples=True)
    assert (abs(epsi4 - ecc) < 1e-11)
