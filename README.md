socc
==============================
[//]: # (Badges)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Actions Build Status](https://github.com/lothian/socc/workflows/CI/badge.svg)](https://github.com/lothian/socc/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lothian/socc/branch/main/graph/badge.svg)](https://codecov.io/gh/lothian/socc/branch/main)

A spin-orbital-based reference CC implementation. Current capabilities include:
  - Spin-orbital CCSD, CCSD(T), and CC3 energies
  - Triples-drivers for various approximate triples methods
  - CCSD densities
  - Spin-orbital CCSD and CC3 dynanmic linear response functions

This repository is currently under development. To do a developmental install, download this repository and type `pip install -e .` in the repository directory.

This package requires the following:
  - [psi4](https://psicode.org)
  - [numpy](https://numpy.org/)
  - [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)

### Author

T. Daniel Crawford

### Copyright

Copyright (c) 2025, T. Daniel Crawford

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
