import os
import numpy as np

# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve

# Contains lots of important functions.
import definition as definition
# Sets up the lattice for the system
import hub_lats as hub

# These also contain various important observable calculators
import harmonic as har_spec
import observable as observable

# Not strictly necessary, but nice to have.
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc


def main_sim(number, nx, u, delta, F0):
    """Number of electrons"""
    # this specifically enforces spin up number being equal to spin down
    nelec = (number, number)

    """number of sites"""
    ny = 0

    """System Parameters"""
    t = 0.52
    U = u * t
    field = 32.9
    a = 4
    cycles = 10


    """these lists get populated with the appropriate expectations"""
    neighbour = []
    energy = []
    doublon_energy = []
    phi_original = []
    # This is just to check the phi reconstruction does what it's supposed to.
    phi_reconstruct = [0., 0.]
    J_field = []
    two_body = []
    D = []
    energy = []

    alt_ham = False

    """used for saving expectations after simulation"""
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude' % (
    nx, cycles, U, t, number, delta, field, F0)

    """class that contains all the essential parameters+scales them. V. IMPORTANT"""
    prop = definition.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
    print('\n')
    print(vars(prop))

    time = cycles

    """This sets initial wavefunction as the ground state"""
    if alt_ham:
        phi0 = 0.000001
        psi_temp = definition.hubbard_alt(prop, phi0)[1].astype(complex)
        h = observable.hamiltonian(hub.create_1e_ham(prop, True), phi0)
    else:
        psi_temp = definition.hubbard(prop)[1].astype(complex)
        h = hub.create_1e_ham(prop, True)


    N = int(time / (prop.freq * delta)) + 1
    print(N)

    """RK4 Method"""
    for k in range(N):
        newtime = k * delta
        psi_temp = evolve.RK4(prop, h, delta, newtime, psi_temp, time)

        definition.progress(N, int(newtime / delta))
        neighbour.append(har_spec.nearest_neighbour_new(prop, h, psi_temp))
        J_field.append(har_spec.J_expectation(prop, h, psi_temp, newtime, time))
        phi_original.append(har_spec.phi(prop, newtime, time))
        energy.append(observable.energy(prop, observable.hamiltonian(h, phi_original[-1]), psi_temp))


    del phi_reconstruct[0:2]
    np.save('./data/original/Jfield' + parameternames, J_field)
    np.save('./data/original/phi' + parameternames, phi_original)
    np.save('./data/original/neighbour' + parameternames, neighbour)
    np.save('./data/original/energy' + parameternames, energy)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "%s" % 1
    main_sim(3, 6, 0, 0.001, 10)
