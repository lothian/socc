import numpy as np
import opt_einsum

def permute_triples(ijkabc, perm_ijk, perm_abc):
    allowed_idx = ['i','j','k','a','b','c']
    idx = {'i':0, 'j':1, 'k':2, 'a':3, 'b':4, 'c':5}

    char_list = list(perm_ijk)
    if char_list[1] != '/':
        raise Exception('String format must be, e.g., i/jk.')
    char = [char_list[0], char_list[2], char_list[3]]
    if set(char).issubset(allowed_idx) is False:
        raise Exception('Only allowed indices are i, j, k or a, b, c.')
    ijk_perm1 = [idx[char[0]], idx[char[1]]]
    ijk_perm2 = [idx[char[0]], idx[char[2]]]

    char_list = list(perm_abc)
    if char_list[1] != '/':
        raise Exception('String format must be, e.g., c/ab.')
    char = [char_list[0], char_list[2], char_list[3]]
    if set(char).issubset(allowed_idx) is False:
        raise Exception('Only allowed indices are i, j, k or a, b, c.')
    abc_perm1 = [idx[char[0]], idx[char[1]]]
    abc_perm2 = [idx[char[0]], idx[char[2]]]

    t3 =  ijkabc \
         -ijkabc.swapaxes(abc_perm1[0],abc_perm1[1]) \
         -ijkabc.swapaxes(abc_perm2[0],abc_perm2[1]) \
         -ijkabc.swapaxes(ijk_perm1[0],ijk_perm1[1]) \
         +ijkabc.swapaxes(ijk_perm1[0],ijk_perm1[1]).swapaxes(abc_perm1[0],abc_perm1[1]) \
         +ijkabc.swapaxes(ijk_perm1[0],ijk_perm1[1]).swapaxes(abc_perm2[0],abc_perm2[1]) \
         -ijkabc.swapaxes(ijk_perm2[0],ijk_perm2[1]) \
         +ijkabc.swapaxes(ijk_perm2[0],ijk_perm2[1]).swapaxes(abc_perm1[0],abc_perm1[1]) \
         +ijkabc.swapaxes(ijk_perm2[0],ijk_perm2[1]).swapaxes(abc_perm2[0],abc_perm2[1])

    return t3

def print_wfn(t1, t2, t3=0):
    no = t1.shape[0]
    nv = t1.shape[1]

    max_print = 5

    t1_amps = -np.abs(t1.flatten())
    t1_idx = np.argsort(t1_amps)
    this_val = 0.0
    num_printed = 0
    for idx in range(len(t1_amps)):
        ia = t1_idx[idx]
        i = ia//nv
        a = ia%nv
        if np.abs(np.abs(this_val)-np.abs(t1[i,a])) > 1e-10 and np.abs(t1[i,a]) > 1e-12 and num_printed < max_print:
            this_val = t1[i,a]
            print("%d %d %20.14f" % (i, a, t1[i,a]))
            num_printed += 1

    t2_amps = -np.abs(t2.flatten())
    t2_idx = np.argsort(t2_amps)
    this_val = 0.0
    num_printed = 0
    for idx in range(len(t2_amps)):
        ijab = t2_idx[idx]
        i = ijab//(no*nv*nv)
        jab = ijab%(no*nv*nv)
        j = jab//(nv*nv)
        ab = jab%(nv*nv)
        a = ab//nv
        b = ab%nv
        if np.abs(np.abs(this_val)-np.abs(t2[i,j,a,b])) > 1e-10 and np.abs(t2[i,j,a,b]) > 1e-12 and num_printed < max_print:
            this_val = t2[i,j,a,b]
            print("%d %d %d %d %20.14f" % (i, j, a, b, t2[i,j,a,b]))
            num_printed += 1

    if t3 != 0:
        t3_amps = -np.abs(t3.flatten())
        t3_idx = np.argsort(t3_amps)
        this_val = 0.0
        num_printed = 0
        for idx in range(len(t3_amps)):
            ijkabc = t3_idx[idx]
            i = ijkabc//(no*no*nv*nv*nv)
            jkabc = ijkabc%(no*no*nv*nv*nv)
            j = jkabc//(no*nv*nv*nv)
            kabc = jkabc%(no*nv*nv*nv)
            k = kabc//(nv*nv*nv)
            abc = kabc%(nv*nv*nv)
            a = abc//(nv*nv)
            bc = abc%(nv*nv)
            b = bc//(nv)
            c = bc%(nv)
            if np.abs(np.abs(this_val)-np.abs(t3[i,j,k,a,b,c])) > 1e-10 and np.abs(t3[i,j,k,a,b,c]) > 1e-12 and num_printed < max_print:
                this_val = t3[i,j,k,a,b,c]
                print("%d %d %d %d %d %d %20.14f" % (i, j, k, a, b, c, t3[i,j,k,a,b,c]))
                num_printed += 1


class helper_diis(object):
    def __init__(self, t1, t2, max_diis):
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()
        self.diis_vals_t1 = [t1.copy()]
        self.diis_vals_t2 = [t2.copy()]

        self.diis_errors = []
        self.diis_size = 0
        self.max_diis = max_diis

    def add_error_vector(self, t1, t2):
        # Add DIIS vectors
        self.diis_vals_t1.append(t1.copy())
        self.diis_vals_t2.append(t2.copy())
        # Add new error vectors
        error_t1 = (self.diis_vals_t1[-1] - self.oldt1).ravel()
        error_t2 = (self.diis_vals_t2[-1] - self.oldt2).ravel()
        self.diis_errors.append(np.concatenate((error_t1, error_t2)))
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

    def extrapolate(self, t1, t2):

        if (self.max_diis == 0):
            return t1, t2

        # Limit size of DIIS vector
        if (len(self.diis_errors) > self.max_diis):
            del self.diis_vals_t1[0]
            del self.diis_vals_t2[0]
            del self.diis_errors[0]

        self.diis_size = len(self.diis_errors)

        # Build error matrix B
        B = np.ones((self.diis_size + 1, self.diis_size + 1)) * -1
        B[-1, -1] = 0

        for n1, e1 in enumerate(self.diis_errors):
            B[n1, n1] = np.dot(e1, e1)
            for n2, e2 in enumerate(self.diis_errors):
                if n1 >= n2:
                    continue
                B[n1, n2] = np.dot(e1, e2)
                B[n2, n1] = B[n1, n2]

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector
        resid = np.zeros(self.diis_size + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        t1 = np.zeros_like(self.oldt1)
        t2 = np.zeros_like(self.oldt2)
        for num in range(self.diis_size):
            t1 += ci[num] * self.diis_vals_t1[num + 1]
            t2 += ci[num] * self.diis_vals_t2[num + 1]

        # Save extrapolated amplitudes to old_t amplitudes
        self.oldt1 = t1.copy()
        self.oldt2 = t2.copy()

        return t1, t2

