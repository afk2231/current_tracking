import numpy as np
import evolve as evolve
import observable as observable
import definition as harmonic
import hub_lats as hub
import harmonic as har_spec
from scipy.integrate import ode
from scipy.interpolate import interp1d


def adaptive_tracking(number, delta, u, F0, ascale):
    unwravel = True

    """Initializing vectors"""
    neighbour = []
    phi_raw = []
    phi_original = []
    phi_unwravel = []
    theta_raw = []
    theta_unwravel = []
    J_field_track_original = []
    J_field_track_unwravel = []

    """loading parameters"""
    nx = 2*number
    ny = 0
    t = 0.52
    U = u * t
    U_track = U
    cycles = 10
    field = 32.9
    scalefactor = 1.0
    a = 4

    """Used for LOADING the expectation you want to track"""
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
        nx, cycles, U, t, number, delta, field, F0)

    """SAVES the tracked simulation."""
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)

    """loading J_field"""
    J_field = np.load('./data/original/Jfield' + parameternames)

    """Sets up the system in which we do tracking. Note that the lattice parameter is scaled by ascale"""
    lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t, F0=F0, a=ascale * a,
                       bc='pbc')
    times = np.linspace(0.0, cycles / lat.freq, len(J_field))
    psi_temp = harmonic.hubbard(lat)[1].astype(complex)
    h = hub.create_1e_ham(lat, True)
    N = int(cycles / (lat.freq * delta)) + 1

    """Interpolates the current to be tracked."""
    J_func = interp1d(times, scalefactor * J_field, fill_value='extrapolate', bounds_error=False, kind='cubic')

    """Loading integrator"""
    r = ode(evolve.integrate_f_track_J).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(lat, h, J_func)

    prevD = np.angle(har_spec.nearest_neighbour_new(lat, h, psi_temp))
    k = 0
    while r.successful() and r.t < cycles/lat.freq:
        r.integrate(r.t + delta)
        psi_temp = r.y
        newtime = r.t
        if k >= 1 and unwravel:
            prevD = neighbour[-1]

        harmonic.progress(N, int(newtime / delta))

        neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
        phi_raw.append(evolve.phi_J_track(lat, newtime, J_func, neighbour[-1]))
        theta_raw.append(np.angle(neighbour[-1]))
        phi_original.append(phi_raw[-1] + theta_raw[-1])
        J_field_track_original.append(har_spec.J_expectation_track(lat, h, psi_temp, phi_original[-1]))
        if unwravel:
            theta_unwravel.append(evolve.angle_fix(neighbour[-1], prevD))
            phi_unwravel.append(phi_raw[-1] + theta_unwravel[-1])
            J_field_track_unwravel.append(har_spec.J_expectation_track(lat, h, psi_temp, phi_unwravel[-1]))

        k += 1

    np.save('./data/tracking/Jfield_original_ap' + newparameternames, J_field_track_original)
    np.save('./data/tracking/theta_raw_ap' + newparameternames, theta_raw)
    np.save('./data/tracking/theta_unwravel_ap' + newparameternames, theta_unwravel)
    np.save('./data/tracking/Jfield_fixed_ap' + newparameternames, J_field_track_unwravel)
    np.save('./data/tracking/phi_raw_ap' + newparameternames, phi_raw)
    np.save('./data/tracking/phi_original_ap' + newparameternames, phi_original)
    np.save('./data/tracking/phifixed_ap' + newparameternames, phi_unwravel)
    np.save('./data/tracking/neighbour_ap' + newparameternames, neighbour)
