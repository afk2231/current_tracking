import newevolve as nev
import evolve as ev
import definition as de
import hub_lats as hub
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import harmonic as ha
import os
import branchfixing as bf
import observable as ob
import matplotlib.pyplot as plt

def work_phi_dot_tracking(F0):
    neighbour = []
    J_field_track_original = []
    phi_original = []
    phi_unwravel = []
    J_field_track_unwravel = []
    stepsize = []
    psi = []
    time = []
    energies = []
    number = 3
    nx = 6
    t_s = 0.52
    U = 0 * t_s
    U_track = U
    cycles = 10
    field = 32.9
    delta = 0.01
    a = 4

    jumping = False
    alt_ham = False

    ascale = 1.0
    scalefactor = 1.0

    """Used for LOADING the expectation you want to track"""
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
        nx, cycles, U, t_s, number, delta, field, F0)

    """SAVES the tracked simulation."""
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t_s, number, delta, field, F0, ascale, scalefactor)

    J_field = np.load('./data/original/Jfield' + parameternames)
    Energy = np.load('./data/original/energy' + parameternames)
    H_dot = np.gradient(Energy, delta)

    lat = de.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t_s, F0=F0, a=ascale * a,
                       bc='pbc')
    times = np.linspace(0.0, cycles / lat.freq, len(J_field))
    N = int(cycles / (lat.freq * delta))

    if alt_ham:
        phi0 = 0.01
        psi_0 = de.hubbard_alt(lat, phi0)[1].astype(complex)
        h = ob.hamiltonian(hub.create_1e_ham(lat, True), phi0)
    else:
        psi_0 = de.hubbard(lat)[1].astype(complex)
        h = hub.create_1e_ham(lat, True)



    """Interpolates the current to be tracked."""
    J_func = interp1d(times, scalefactor * J_field, fill_value='extrapolate', bounds_error=False, kind='cubic')
    e_dot_func = interp1d(times, H_dot, fill_value='extrapolate', bounds_error=False, kind='cubic')

    #plt.subplot(311)
    #plt.plot(times, H_dot)
    #plt.subplot(312)
    #plt.plot(times, e_dot_func(times))
    #plt.subplot(313)
    #plt.plot(times, e_dot_func(times)-H_dot)
    #plt.show()

    #plt.subplot(311)
    #plt.plot(times, J_field)
    #plt.subplot(312)
    #plt.plot(times, J_func(times))
    #plt.subplot(313)
    #plt.plot(times, J_func(times)-J_field)
    #plt.show()


    plt.plot(times, J_func(times), linestyle="solid")
    plt.plot(times, e_dot_func(times), linestyle="dashed")
    #plt.plot(times[20: -20], (e_dot_func(times)/J_func(times))[20:-20], linestyle="dotted")
    plt.show()

    phi_0 = np.complex(ev.phi_J_track(lat, 0, J_func, ha.nearest_neighbour_new(lat, h, psi_0))
                       + np.angle(ha.nearest_neighbour_new(lat, h, psi_0)))

    """initialize y = (phi, psi)"""
    y_i = [phi_0.real, np.abs(ha.nearest_neighbour_new(lat, h, psi_0)), np.angle(ha.nearest_neighbour_new(lat, h, psi_0))]
    for j in range(len(psi_0)):
        y_i.append(psi_0[j].real)
    for j in range(len(psi_0)):
        y_i.append(psi_0[j].imag)
    y_init = y_i

    #print(psi_temp)
    #print(y)
    delta_0 = delta
    """integrate"""
    for n in range(10, N - 10):
        t = n * delta
        time.append(t * lat.freq)
        y_i = nev.prelim_work_coupled_RK4(time[-1] / lat.freq, y_i, lat, h, J_func, e_dot_func, delta)
        stepsize.append(delta)
        de.progress(N, n)
        phi_original.append(y_i[0])
        psi.append(nev.recombine_psi(y_i))
        J_field_track_original.append(ha.J_expectation_track(lat, h, psi[-1], phi_original[-1]))
        energies.append(ob.energy(lat, nev.coupled_h(y_i, h), psi[-1]))


    np.save('./data/phi_dot_tracking/phi_work' + newparameternames, phi_original)
    np.save('./data/phi_dot_tracking/J_field_work' + newparameternames, J_field_track_original)
    np.save('./data/phi_dot_tracking/neighbour_work' + newparameternames, neighbour)
    np.save('./data/phi_dot_tracking/time_work' + newparameternames, time)
    np.save('./data/phi_dot_tracking/stepsize_work' + newparameternames, stepsize)
    np.save('./data/phi_dot_tracking/energy_work'+ newparameternames, energies)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "%s" % 3
    work_phi_dot_tracking(3)
