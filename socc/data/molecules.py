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

# Chiral H2O2 for optical rotation test
h2o2 = """
O
O 1 1.39
H 1 0.94 2 102.3
H 2 0.94 1 102.3 3 -50.0
symmetry c1
"""

moldict = {}
moldict["H2O"] = h2o
moldict["CH2"] = ch2
moldict["H2O_Pycc"] = h2o_pycc
moldict["(H2)_2"] = h2_2
moldict["H2O2"] = h2o2
