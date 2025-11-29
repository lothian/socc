import numpy as np
from opt_einsum import contract
from .utils import print_wfn

def t3c_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo, omega=0.0, WithDenom=True):

    abc = contract('ad,bcd->abc', t2[i,j], Wvvvo[:,:,:,k])
    abc -= contract('ad,bcd->abc', t2[k,j], Wvvvo[:,:,:,i])
    abc -= contract('ad,bcd->abc', t2[i,k], Wvvvo[:,:,:,j])
    t3 = abc - abc.swapaxes(0,1) - abc.swapaxes(0,2)

    abc = contract('lab,lc->abc', t2[i], Wovoo[:,:,j,k])
    abc -= contract('lab,lc->abc', t2[j], Wovoo[:,:,i,k])
    abc -= contract('lab,lc->abc', t2[k], Wovoo[:,:,j,i])
    t3 -= abc - abc.swapaxes(0,2) - abc.swapaxes(1,2)

    if WithDenom is True:
        denom = np.zeros_like(t3)
        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        denom -= vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir
        denom += occ[i] + occ[j] + occ[k]
        denom += omega

        return t3/denom
    else:
        return t3


def t3c_abc(o, v, a, b, c, t2, F, Wvvvo, Wovoo, omega=0.0, WithDenom=True):

    ijk = contract('ijd,dk->ijk', t2[:,:,a], Wvvvo[b,c])
    ijk -= contract('ijd,dk->ijk', t2[:,:,b], Wvvvo[a,c])
    ijk -= contract('ijd,dk->ijk', t2[:,:,c], Wvvvo[b,a])
    t3 = ijk - ijk.swapaxes(0,2) - ijk.swapaxes(1,2)

    ijk = contract('il,ljk->ijk', t2[:,:,a,b], Wovoo[:,c])
    ijk -= contract('il,ljk->ijk', t2[:,:,c,b], Wovoo[:,a])
    ijk -= contract('il,ljk->ijk', t2[:,:,a,c], Wovoo[:,b])
    t3 -= ijk - ijk.swapaxes(0,1) - ijk.swapaxes(0,2)

    if WithDenom is True:
        denom = np.zeros_like(t3)
        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        denom += occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ
        denom -= vir[a] + vir[b] + vir[c]
        denom += omega

        return t3/denom
    else:
        return t3


def t_viking_ijk(o, v, t1, t2, F, ERI):

    x1 = np.zeros_like(t1)
    x2 = np.zeros_like(t2)
    no = x1.shape[0]
    nv = x1.shape[1]

    t3_full = np.zeros((no, no, no, nv, nv, nv))

    for i in range(no):
        for j in range(no):
            for k in range(no):

                t3 = t3c_ijk(o, v, i, j, k, t2, F, ERI[v,v,v,o], ERI[o,v,o,o])
                t3_full[i,j,k] = t3

                x1[i] += (1/4)*contract('bc,abc->a', ERI[j,k,v,v], t3)
                tmp = (1/2)*contract('dbc,abc->ad', ERI[v,k,v,v], t3)
                x2[i,j] += tmp - tmp.swapaxes(0,1)
                for l in range(no):
                    tmp = (1/2)*contract('c,abc->ab', ERI[j,k,l,v], t3)
                    x2[i,l] -= tmp
                    x2[l,i] += tmp

    et = contract('ia,ia->', t1, x1) + (1/4)*contract('ijab,ijab->', t2, x2)

    return et


def t_viking_abc(o, v, t1, t2, F, ERI):

    x1 = np.zeros_like(t1.T)
    x2 = np.zeros_like(t2.T)
    nv = x1.shape[0]
    no = x1.shape[1]

    for a in range(nv):
        for b in range(nv):
            for c in range(nv):

                t3 = t3c_abc(o, v, a, b, c, t2, F, ERI[v,v,v,o], ERI[o,v,o,o])

                x1[a] += (1/4)*contract('jk,ijk->i', ERI[o,o,b+no,c+no], t3)
                tmp = -(1/2)*contract('jkl,ijk->il', ERI[o,o,o,c+no], t3)
                x2[a,b] += tmp - tmp.swapaxes(0,1)
                for d in range(nv):
                    tmp = (1/2)*contract('k,ijk->ij', ERI[d+no,o,b+no,c+no], t3)
                    x2[a,d] += tmp
                    x2[d,a] -= tmp

    et = contract('ia,ia->', t1, x1.T) + (1/4)*contract('ijab,ijab->', t2, x2.T)

    return et

def l3_ijk(o, v, i, j, k, l1, l2, F, Fov, Woovv, Wvovv, Wooov, WithDenom=True):
    abc = contract('ad,dbc->abc', l2[i,j], Wvovv[:,k,:,:])
    abc -= contract('ad,dbc->abc', l2[k,j], Wvovv[:,i,:,:])
    abc -= contract('ad,dbc->abc', l2[i,k], Wvovv[:,j,:,:])
    l3 = abc - abc.swapaxes(0,1) - abc.swapaxes(0,2)

    abc = contract('lab,lc->abc', l2[j], Wooov[i,k])
    abc -= contract('lab,lc->abc', l2[i], Wooov[j,k])
    abc += contract('lab,lc->abc', l2[k], Wooov[j,i])
    l3 += abc - abc.swapaxes(0,2) - abc.swapaxes(1,2)

    abc = contract('a,bc->abc', l1[i], Woovv[j,k]) + contract('a,bc->abc', Fov[i], l2[j,k])
    abc -= contract('a,bc->abc', l1[j], Woovv[i,k]) + contract('a,bc->abc', Fov[j], l2[i,k])
    abc -= contract('a,bc->abc', l1[k], Woovv[j,i]) + contract('a,bc->abc', Fov[k], l2[j,i])
    l3 += abc - abc.swapaxes(0,1) - abc.swapaxes(0,2)

    if WithDenom is True:
        denom = np.zeros_like(l3)
        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        denom -= vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir
        denom += occ[i] + occ[j] + occ[k]

        return l3/denom
    else:
        return l3


def l3_abc(o, v, a, b, c, l1, l2, F, Fov, Woovv, Wvovv, Wooov, WithDenom=True):
    ijk = contract('ijd,dk->ijk', l2[:,:,a,:], Wvovv[:,:,b,c])
    ijk -= contract('ijd,dk->ijk', l2[:,:,b,:], Wvovv[:,:,a,c])
    ijk -= contract('ijd,dk->ijk', l2[:,:,c,:], Wvovv[:,:,b,a])
    l3 = ijk - ijk.swapaxes(0,2) - ijk.swapaxes(1,2)

    ijk = contract('il,jkl->ijk', l2[:,:,a,b], Wooov[:,:,:,c])
    ijk -= contract('il,jkl->ijk', l2[:,:,c,b], Wooov[:,:,:,a])
    ijk -= contract('il,jkl->ijk', l2[:,:,a,c], Wooov[:,:,:,b])
    l3 -= ijk - ijk.swapaxes(0,1) - ijk.swapaxes(0,2)

    ijk = contract('i,jk->ijk', l1[:,a], Woovv[:,:,b,c]) + contract('i,jk->ijk', Fov[:,a], l2[:,:,b,c])
    ijk -= contract('i,jk->ijk', l1[:,b], Woovv[:,:,a,c]) + contract('i,jk->ijk', Fov[:,b], l2[:,:,a,c])
    ijk -= contract('i,jk->ijk', l1[:,c], Woovv[:,:,b,a]) + contract('i,jk->ijk', Fov[:,c], l2[:,:,b,a])
    l3 += ijk - ijk.swapaxes(0,1) - ijk.swapaxes(0,2)

    if WithDenom is True:
        denom = np.zeros_like(l3)
        occ = np.diag(F)[o]
        vir = np.diag(F)[v]
        denom += occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ
        denom -= vir[a] + vir[b] + vir[c]

        return l3/denom
    else:
        return l3


def X3_ijk(o, v, i, j, k, t2, F, pert, X2, Wvvvo, Wovoo, Zvvvo, Zovoo, omega):

    occ = np.diag(F)[o]
    vir = np.diag(F)[v]

    # <mu3|[ABAR,T3]|0> --> X3
    t3 = t3c_ijk(o, v, i, j, k, t2, F, Wvvvo, Wovoo)
    tmp = contract('abc,dc->abd', t3, pert.Avv)
    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
    denom = np.zeros_like(t3)
    denom -= vir.reshape(-1,1,1) + vir.reshape(-1,1) + vir
    denom += occ[i] + occ[j] + occ[k]
    denom += omega
    x3 = x3/denom

    # <mu3|[[ABAR,T2],T2]|0> + <mu3|[[H^,T2,X1]|0> --> X3
    x3 += t3c_ijk(o, v, i, j, k, t2, F, Zvvvo, Zovoo, omega)

    # <mu3|[H^,X2]|0> --> X3
    x3 += t3c_ijk(o, v, i, j, k, X2, F, Wvvvo, Wovoo, omega)

    return x3


def X3_abc(o, v, a, b, c, t2, F, pert, Wvvvo, Wovoo, omega):

    occ = np.diag(F)[o]
    vir = np.diag(F)[v]

    # <mu3|[ABAR,T3]|0> --> X3
    t3 = t3c_abc(o, v, a, b, c, t2, F, Wvvvo, Wovoo)
    tmp = -contract('ijk,kl->ijl', t3, pert.Aoo)
    x3 = tmp - tmp.swapaxes(0,2) - tmp.swapaxes(1,2)
    denom = np.zeros_like(t3)
    denom += occ.reshape(-1,1,1) + occ.reshape(-1,1) + occ
    denom -= vir[a] + vir[b] + vir[c]
    denom += omega
    x3 = x3/denom

    return x3
