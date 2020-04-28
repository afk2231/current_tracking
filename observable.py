import numpy as np
from pyscf import fci
import definition as harmonic
import evolve as evolve
import des_cre as dc


def DHP(lat, psi):
    # psi = np.reshape(psi,(fci.cistring.num_strings(lat.nsites,lat.nup),fci.cistring.num_strings(lat.nsites,lat.ndown)))
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(lat.nsites, lat.nup), fci.cistring.num_strings(lat.nsites, lat.ndown)))
    D = 0.
    for i in range(lat.nsites):
        D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [i, i, i, i], [1, 0, 1, 0],
                                            [1, 1, 0, 0])
    # return D/lat.nsites
    return D / lat.nsites


def single_operator(lat, psi):
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(lat.nsites, lat.nup), fci.cistring.num_strings(lat.nsites, lat.ndown)))
    D = 0.
    civec = psi
    for j in [0, 1]:
        for i in range(lat.nsites - 1):
            D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [i], [1], [j])
    return D


def spin(lat, psi):
    eta = 0.
    for i in range(lat.nsites):
        j = (i + 1) % lat.nsites
        # abba
        eta += 0.5 * harmonic.compute_inner_product(psi, lat.ne, (lat.nup, lat.ndown), [i, i, j, j], [1, 0, 1, 0],
                                                    [1, 0, 0, 1])
        # baab
        eta += 0.5 * harmonic.compute_inner_product(psi, lat.ne, (lat.nup, lat.ndown), [i, i, j, j], [1, 0, 1, 0],
                                                    [0, 1, 1, 0])
        # aaaa
        eta += 0.25 * harmonic.compute_inner_product(psi, lat.ne, (lat.nup, lat.ndown), [i, i, j, j], [1, 0, 1, 0],
                                                     [1, 1, 1, 1])
        # bbbb
        eta += 0.25 * harmonic.compute_inner_product(psi, lat.ne, (lat.nup, lat.ndown), [i, i, j, j], [1, 0, 1, 0],
                                                     [0, 0, 0, 0])
        # aabb
        eta -= 0.25 * harmonic.compute_inner_product(psi, lat.ne, (lat.nup, lat.ndown), [i, i, j, j], [1, 0, 1, 0],
                                                     [1, 1, 0, 0])
        # bbaa
        eta -= 0.25 * harmonic.compute_inner_product(psi, lat.ne, (lat.nup, lat.ndown), [i, i, j, j], [1, 0, 1, 0],
                                                     [0, 0, 1, 1])
    return eta.real / lat.nsites


def overlap(lat, initial, psi):
    fid = np.dot(initial.conj(), psi)
    fid = abs(fid)
    return (fid ** 2, fid)


def energy(lat, h, psi):
    E = lat.t*np.dot(psi.conj(), evolve.f(lat, h, psi))
    if E.imag < 10e-11:
        E = E.real
    # print(E)
    return E

def number(lat, h, psi):

    N = np.dot(psi.conj(), )


def hamiltonian(h, phi):
    h_forwards = np.triu(h)
    h_forwards[0, -1] = 0.0
    h_forwards[-1, 0] = h[-1, 0]
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    return np.exp(1.j * phi) * h_forwards + np.exp(-1.j * phi) * h_backwards
