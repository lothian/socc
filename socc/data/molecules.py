# Cartesian coordinates for H2O for comparison to CFOUR
h2o = """
0 1
O 0.000000000000000   0.000000000000000   0.143225857166674
H 0.000000000000000  -1.638037301628121  -1.136549142277225
H 0.000000000000000   1.638037301628121  -1.136549142277225
symmetry c1
units bohr
"""

# Cartesian coordinates for CH2 for comparison to CFOUR
ch2 = """
0 3
C 0.000000000000000   0.000000000000000   0.184049869562229
H 0.000000000000000  -1.638037301628121  -1.095725129881670
H 0.000000000000000   1.638037301628121  -1.095725129881670
symmetry c1
units bohr
"""

# H2O test case from Pycc
h2o_pycc = """
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
"""

# Chiral (H2)2 from Pycc for optical rotation test
h2_2 = """
H
H 1 0.75
H 2 1.5 1 90.0
H 3 0.75 2 90.0 1 60.0
symmetry c1
"""

# Chiral H2O2 from Dalton for optical rotation test
h2o2 = """
O   1.3133596569   0.0000000000  -0.0932359644
O  -1.3133596569  -0.0000000000  -0.0932359644
H   1.6917745981   0.7334825768   1.4797224976
H  -1.6917745981  -0.7334825768   1.4797224976
symmetry c1
units bohr
"""

moldict = {}
moldict["H2O"] = h2o
moldict["CH2"] = ch2
moldict["H2O_Pycc"] = h2o_pycc
moldict["(H2)_2"] = h2_2
moldict["H2O2"] = h2o2
