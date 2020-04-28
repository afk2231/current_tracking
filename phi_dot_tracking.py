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

def phi_dot_tracking(F0):
    neighbour = []
    J_field_track_original = []
    phi_original = []
    y_dot = [0]
    phi_unwravel = []
    J_field_track_unwravel = []
    stepsize = []
    psi = []
    time = [0]
    timed = []
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

    jumping = True
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
    grad_J_field = np.gradient(J_field, delta)

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
    J_dot_func = interp1d(times, scalefactor * grad_J_field, fill_value='extrapolate', bounds_error=False, kind='cubic')

    phi_0 = np.complex(ev.phi_J_track(lat, 0, J_func, ha.nearest_neighbour_new(lat, h, psi_0))
                       + np.angle(ha.nearest_neighbour_new(lat, h, psi_0)))

    """initialize y = (phi, psi)"""
    y_i = [phi_0.real, np.abs(ha.nearest_neighbour_new(lat, h, psi_0)), np.angle(ha.nearest_neighbour_new(lat, h, psi_0))]
    for j in range(len(psi_0)):
        y_i.append(psi_0[j].real)
    for j in range(len(psi_0)):
        y_i.append(psi_0[j].imag)

    bj = 0
    n = 0
    ndelta = delta
    r = []


    if jumping:
        while n + bj*(ndelta/delta) < N:
            t = (n * delta + bj * ndelta)
            de.progress(N, n+int(bj*(ndelta/delta)))
            phi_original.append(y_i[0])
            psi.append(nev.recombine_psi(y_i))
            #print(np.dot(psi[-1].conj(), psi[-1]))
            J_field_track_original.append(ha.J_expectation_track(lat, h, psi[-1], phi_original[-1]))
            energies.append(ob.energy(lat, nev.coupled_h(y_i, h), psi[-1]))
            cond = (0.9 * np.pi/2 < abs(phi_original[-1]) < 1.1 * np.pi/2)
            if cond:
                bj = bj + 1
                y_i = nev.coupled_RK4_alt(t, y_i, lat, h, J_func, ndelta, y_dot, r, time)
                stepsize.append(ndelta)
                time.append(t * lat.freq)
                r.append(ndelta / stepsize[-1])
                check = 1
                #print("ping")
            else:
                y_i = nev.coupled_RK4(t, y_i, lat, h, J_dot_func, delta, y_dot)
                stepsize.append(delta)
                time.append(t * lat.freq)
                r.append(ndelta / stepsize[-1])
                n += 1
                check = 0
                if time[-1] < time[-2]:
                    exit("fuck")

        time.pop(0)
    else:
        y_dot = integrate.RK45(lambda t, y: nev.coupled_evolve(t, y, lat, h, J_dot_func), delta, y_i, N,
                               vectorized=True,
                               first_step=delta, max_step=delta)
        #print(psi_temp)
        #print(y)
        delta_0 = delta

        while y_dot.t < cycles/lat.freq and y_dot.status == 'running':
            de.progress(N, int(y_dot.t / lat.freq))
            y_dot.step()
            time.append(y_dot.t * lat.freq)
            stepsize.append(y_dot.step_size)
            phi_original.append(y_dot.y[0])
            psi.append(nev.recombine_psi(y_dot.y))
            J_field_track_original.append(ha.J_expectation_track(lat, h, psi[-1], phi_original[-1]))
            energies.append(ob.energy(lat, nev.coupled_h(y_dot.y, h), psi[-1]))

            neighbour.append(ha.nearest_neighbour_new(lat, h, psi[-1]))
        time.pop(0)

    np.save('./data/phi_dot_tracking/phi' + newparameternames, phi_original)
    np.save('./data/phi_dot_tracking/J_field' + newparameternames, J_field_track_original)
    np.save('./data/phi_dot_tracking/neighbour' + newparameternames, neighbour)
    np.save('./data/phi_dot_tracking/time' + newparameternames, time)
    np.save('./data/phi_dot_tracking/stepsize' + newparameternames, stepsize)
    np.save('./data/phi_dot_tracking/energy'+ newparameternames, energies)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "%s" % 3
    phi_dot_tracking(3)