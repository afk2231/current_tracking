import newevolve as nev
import evolve as ev
import definition as de
import hub_lats as hub
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import harmonic as ha
import os
os.environ["OMP_NUM_THREADS"] = "%s" % 5

if __name__ == "__main__":
    neighbour = []
    J_field_track_original = []
    phi_original = []
    phi_unwravel = []
    J_field_track_unwravel = []
    psi = []
    time = []
    number = 3
    nx = 2*number
    nelec = (number, number)
    ny = 0
    t_s = 0.52
    U = 0 * t_s
    U_track = U
    cycles = 10
    field = 32.9
    F0 = 3
    delta = 0.01
    a = 4

    ascale = 1.0
    scalefactor = 1.0

    """Used for LOADING the expectation you want to track"""
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
        nx, cycles, U, t_s, number, delta, field, F0)

    """SAVES the tracked simulation."""
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t_s, number, delta, field, F0, ascale, scalefactor)

    J_field = np.load('./data/original/Jfield' + parameternames)
    grad_J_field = np.gradient(J_field, delta)

    lat = de.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t_s, F0=F0, a=ascale * a,
                       bc='pbc')
    times = np.linspace(0.0, cycles / lat.freq, len(J_field))
    N = int(cycles / (lat.freq * delta))

    psi_0 = de.hubbard(lat)[1].astype(complex)
    prevpsi = psi_0
    h = hub.create_1e_ham(lat, True)



    """Interpolates the current to be tracked."""
    J_func = interp1d(times, scalefactor * J_field, bounds_error=False, kind='cubic')
    J_dot_func = interp1d(times, scalefactor * grad_J_field, bounds_error=False, kind='cubic')

    phi_0 = np.complex(ev.phi_J_track(lat, 0, J_func, ha.nearest_neighbour_new(lat, h, psi_0))
                       + np.angle(ha.nearest_neighbour_new(lat, h, psi_0)))

    """initialize y = (phi, psi)"""
    y_i = [phi_0.real, np.abs(ha.nearest_neighbour_new(lat, h, psi_0)), np.angle(ha.nearest_neighbour_new(lat, h, psi_0))]
    for j in range(len(psi_0)):
        y_i.append(psi_0[j].real)
    for j in range(len(psi_0)):
        y_i.append(psi_0[j].imag)

    y_dot = integrate.LSODA(lambda t, y: nev.coupled_evolve(t, y, lat, h, J_dot_func), delta, y_i, N,
                       vectorized=True, first_step=delta)
    #print(psi_temp)
    #print(y)
    delta_0 = delta
    """integrate"""
    while y_dot.t < cycles/lat.freq and y_dot.status == 'running':
        de.progress(N, int(y_dot.t / lat.freq))
        y_dot.step()
        time.append(y_dot.t * lat.freq)
        phi_original.append(y_dot.y[0])
        psi.append(nev.recombine_psi(y_dot.y))
        J_field_track_original.append(ha.J_expectation_track(lat, h, psi[-1], phi_original[-1]))

        neighbour.append(ha.nearest_neighbour_new(lat, h, psi[-1]))

    np.save('./data/phi_dot_tracking/phi_LSODA' + newparameternames, phi_original)
    np.save('./data/phi_dot_tracking/J_field_LSODA' + newparameternames, J_field_track_original)
    np.save('./data/phi_dot_tracking/neighbour_LSODA' + newparameternames, neighbour)
    np.save('./data/phi_dot_tracking/time_LSODA' + newparameternames, time)
