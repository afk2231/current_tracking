import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import evolve as evolve
import observable as observable
import definition as harmonic
import hub_lats as hub
import harmonic as har_spec
from matplotlib import cm as cm
from scipy.integrate import ode
from scipy.interpolate import interp1d


# NOTE: time is inputted and plotted in terms of cycles, but the actual propagation happens in 'normal' time

# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic
def main_track(number, nx, u, delta, ascale, F0):
    unwravel = True
    alt_ham = False
    neighbour = []
    phi_raw = []
    phi_original = []
    phi_unwravel = []
    theta_raw = []
    theta_unwravel = []
    phi_reconstruct = [0., 0.]
    boundary_1 = []
    boundary_2 = []
    two_body = []
    energies = []
    two_body_old = []
    error = []
    J_field_track_original = []
    J_field_track_unwravel = []
    D_track = []

    t = 0.52
    # t=1.91
    # t=1
    """U is the the ORIGINAL data you want to track"""
    U = u * t

    """U_track is the NEW system parameter you want to do tracking in"""
    U_track = U
    cycles = 10
    field = 32.9
    # field=25
    a = 4

    # This scales the lattice parameter

    # this scales the input current.
    scalefactor = 1.0

    """Used for LOADING the expectation you want to track"""
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)

    """SAVES the tracked simulation."""
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)

    J_field = np.load('./data/original/Jfield' + parameternames)
    # D=np.load('./data/original/double'+parameternames)
    # delta=0.01
    # lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
    time = cycles

    # times = np.linspace(0.0, cycles/lat.freq, len(J_field))
    # times = np.linspace(0.0, cycles, len(D))

    """Sets up the system in which we do tracking. Note that the lattice parameter is scaled by ascale"""
    lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t, F0=F0, a=ascale * a,
                       bc='pbc')
    times = np.linspace(0.0, cycles / lat.freq, len(J_field))
    # times = np.linspace(0.0, cycles, len(D))
    print('\n')
    print(vars(lat))
    if alt_ham:
        phi0 = 0.000001
        psi_temp = harmonic.hubbard_alt(lat, phi0)[1].astype(complex)
        h = observable.hamiltonian(hub.create_1e_ham(lat, True), phi0)
    else:
        psi_temp = harmonic.hubbard(lat)[1].astype(complex)
        h = hub.create_1e_ham(lat, True)

    N = int(cycles / (lat.freq * delta)) + 1

    """Interpolates the current to be tracked."""
    J_func = interp1d(times, scalefactor * J_field, fill_value='extrapolate', bounds_error=False, kind='cubic')
    # D_func = interp1d(times, np.gradient(D,delta/(lat.freq)), fill_value=0, bounds_error=False, kind='cubic')

    prop = lat
    # r = ode(evolve.integrate_f_track_J).set_integrator('zvode', method='bdf')
    # r = ode(evolve.integrate_f_track_D).set_integrator('zvode', method='bdf')

    # set which observable to track

    # """Set the ode parameters, including the current to be tracked"""
    # r.set_initial_value(psi_temp, 0).set_f_params(lat,h,J_func)
    # r.set_initial_value(psi_temp, 0).set_f_params(lat,h,D_func)
    prevD = np.angle(har_spec.nearest_neighbour_new(prop, h, psi_temp))
    for k in range(N):
        harmonic.progress(N, k)
        newtime = k * delta
        # if fixing == True:
        # newtime = (k/2)*delta
        # else:

        if k >= 1 and unwravel:
            prevD = neighbour[-1]
            # print(neighbour[-1])
            # add to expectations

        neighbour.append(har_spec.nearest_neighbour_new(prop, h, psi_temp))
        two_body.append(har_spec.two_body(prop, h, psi_temp.real, psi_temp.imag))
        psi_temp = evolve.RK4_J_track(prop, h, delta, newtime, J_func, neighbour[-1], psi_temp)

        # tracking current
        phi_raw.append(evolve.phi_J_track(prop, newtime, J_func, neighbour[-1]))
        theta_raw.append(np.angle(neighbour[-1]))
        phi_original.append(phi_raw[-1] + theta_raw[-1])
        J_field_track_original.append(har_spec.J_expectation_track(prop, h, psi_temp, phi_original[-1]))
        D_track.append(observable.DHP(prop, psi_temp))
        energies.append(observable.energy(prop, observable.hamiltonian(h, phi_original[-1]), psi_temp))
        if unwravel:
            theta_unwravel.append(evolve.angle_fix(neighbour[-1], prevD))
            phi_unwravel.append(phi_raw[-1] + theta_unwravel[-1])
            J_field_track_unwravel.append(har_spec.J_expectation_track(prop, h, psi_temp, phi_unwravel[-1]))


        # diff = (psi_temp - oldpsi) / delta
        # newerror = np.linalg.norm(diff + 1j * psierror)
        # error.append(newerror)
    print('Current tracking finished')
    del phi_reconstruct[0:2]
    """Note that this is now saved with the LOADED parameters, not the ones used in tracking"""
    np.save('./data/tracking/double' + newparameternames, D_track)
    np.save('./data/tracking/Jfield_original' + newparameternames, J_field_track_original)
    np.save('./data/tracking/theta_raw' + newparameternames, theta_raw)
    np.save('./data/tracking/theta_unwravel' + newparameternames, theta_unwravel)
    np.save('./data/tracking/Jfield_fixed' + newparameternames, J_field_track_unwravel)
    np.save('./data/tracking/phi_raw' + newparameternames, phi_raw)
    np.save('./data/tracking/phi_original' + newparameternames, phi_original)
    np.save('./data/tracking/phifixed' + newparameternames, phi_unwravel)
    np.save('./data/tracking/neighbour' + newparameternames, neighbour)
    np.save('./data/tracking/twobody' + newparameternames, two_body)
    np.save('./data/tracking/energies' + newparameternames, energies)


def main_tracking_phi(number, nx, u, delta, ascale, F0):
    neighbour_phi = []
    two_body_phi = []
    phi_raw_phi = []
    theta_raw_phi = []
    phi_original_phi = []
    J_field_track_original_phi = []
    D_track_phi = []
    theta_unwravel_phi = []
    phi_raw_unwravel_phi = []
    J_field_track_unwravel_phi = []
    phi_unwravel_phi = []

    unwravel = True
    phi_reconstruct = [0., 0.]

    nelec = (number, number)
    ny = 0
    t = 0.52
    # t=1.91
    # t=1
    """U is the the ORIGINAL data you want to track"""
    U = u * t

    """U_track is the NEW system parameter you want to do tracking in"""
    U_track = U
    cycles = 10
    field = 32.9
    # field=25
    a = 4


    # this scales the input current.
    scalefactor = 1.0

    """Used for LOADING the expectation you want to track"""
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
        nx, cycles, U, t, number, delta, field, F0)

    """SAVES the tracked simulation."""
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)

    phi_funct = np.load('./data/tracking/phi_original' + newparameternames)
    # D=np.load('./data/original/double'+parameternames)
    # delta=0.01
    # lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
    time = cycles

    # times = np.linspace(0.0, cycles/lat.freq, len(J_field))
    # times = np.linspace(0.0, cycles, len(D))

    """Sets up the system in which we do tracking. Note that the lattice parameter is scaled by ascale"""
    lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t, F0=F0, a=ascale * a,
                       bc='pbc')
    times = np.linspace(0.0, cycles / lat.freq, len(phi_funct))
    # times = np.linspace(0.0, cycles, len(D))
    print('\n')
    print(vars(lat))
    psi_phi_temp = harmonic.hubbard(lat)[1].astype(complex)
    h = hub.create_1e_ham(lat, True)

    N = int(cycles / (lat.freq * delta)) + 1

    """Interpolates the current to be tracked."""
    phi_cut = interp1d(times, phi_funct, fill_value='extrapolate', bounds_error=False, kind='cubic')
    # D_func = interp1d(times, np.gradient(D,delta/(lat.freq)), fill_value=0, bounds_error=False, kind='cubic')

    prop = lat
    prevD = np.angle(har_spec.nearest_neighbour_new(prop, h, psi_phi_temp))
    for k in range(N):
        harmonic.progress(N, k)
        newtime = k * delta
        # if fixing == True:
        # newtime = (k/2)*delta
        # else:

        if k >= 1 and unwravel:
            prevD = neighbour_phi[-1]
            # print(neighbour[-1])
            # add to expectations

        neighbour_phi.append(har_spec.nearest_neighbour_new(prop, h, psi_phi_temp))
        two_body_phi.append(har_spec.two_body(prop, h, psi_phi_temp.real, psi_phi_temp.imag))
        psi_phi_temp = evolve.RK4_J_cutfreqs_RK4(prop, h, newtime, delta, phi_cut, psi_phi_temp)

        # tracking current
        J_field_track_original_phi.append(har_spec.J_expectation_track(prop, h, psi_phi_temp, phi_cut(newtime)))
        phi_original_phi.append(evolve.phi_phi_track(prop, J_field_track_original_phi[-1], neighbour_phi[-1]))
        theta_raw_phi.append(np.angle(neighbour_phi[-1]))
        phi_raw_phi.append(phi_original_phi[-1] - theta_raw_phi[-1])

        D_track_phi.append(observable.DHP(prop, psi_phi_temp))
        if unwravel:
            J_field_track_unwravel_phi.append(har_spec.J_expectation_track(prop, h, psi_phi_temp, phi_cut(newtime)))
            phi_unwravel_phi.append(evolve.phi_phi_track(prop, J_field_track_unwravel_phi[-1], neighbour_phi[-1]))
            theta_unwravel_phi.append(evolve.angle_fix(neighbour_phi[-1], prevD))
            phi_raw_unwravel_phi.append(phi_unwravel_phi[-1] - theta_unwravel_phi[-1])


    print('Phi tracking finished')
    del phi_reconstruct[0:2]

    np.save('./data/tracking/double_phi' + newparameternames, D_track_phi)
    np.save('./data/tracking/Jfield_original_phi' + newparameternames, J_field_track_original_phi)
    np.save('./data/tracking/theta_raw_phi' + newparameternames, theta_raw_phi)
    np.save('./data/tracking/theta_unwravel_phi' + newparameternames, theta_unwravel_phi)
    np.save('./data/tracking/Jfield_fixed_phi' + newparameternames, J_field_track_unwravel_phi)
    np.save('./data/tracking/phi_original_phi' + newparameternames, phi_original_phi)
    np.save('./data/tracking/phi_raw_phi' + newparameternames, phi_raw_phi)
    np.save('./data/tracking/phifixed_phi' + newparameternames, phi_raw_unwravel_phi)
    np.save('./data/tracking/neighbour_phi' + newparameternames, neighbour_phi)
    np.save('./data/tracking/twobody_phi' + newparameternames, two_body_phi)


# plot_observables(lat, delta=0.02, time=5., K=.1)
# spectra(lat, initial=None, delta=delta, time=cycles, method='welch', min_spec=7, max_harm=50, gabor='fL')
