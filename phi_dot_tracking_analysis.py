import numpy as np
import matplotlib.pyplot as plt
import definition as hams
from scipy.interpolate import interp1d

number = 3
nelec = (number, number)
nx = 6
ny = 0
t = 0.52
U = 0 * t
U_track = U
delta = 0.01
cycles = 10
field = 32.9
F0 = 10
a = 4
scalefactor = 1.0
ascale = 1.0
F02 = 3

phi_track = []
J_field_track = []

Tracking = True
track_lsoda = False
track_RK4 = False

prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')

parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
parameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F02)

J_field = np.load('./data/original/Jfield' + parameternames)
phi_original = np.load('./data/original/phi' + parameternames)
J_field2 = np.load('./data/original/Jfield' + parameternames2)
phi_original2 = np.load('./data/original/phi' + parameternames2)
neighbour = np.load('./data/original/neighbour' + parameternames)
energy = np.load('./data/original/energy' + parameternames)
times = np.linspace(0.0, cycles, len(J_field))

if Tracking:
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)
    newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F02, ascale, scalefactor)
    imagefilenames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.png' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)
    imagefilenames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.png' % (
        nx, cycles, U, t, number, delta, field, F02, ascale, scalefactor)
    if track_lsoda:
        J_field_track = np.load('./data/phi_dot_tracking/J_field_LSODA' + newparameternames) / scalefactor
        neighbour_track = np.load('./data/phi_dot_tracking/neighbour_LSODA' + newparameternames)
        phi_track = np.load('./data/phi_dot_tracking/phi_LSODA' + newparameternames)

        t_track = np.load('./data/phi_dot_tracking/time_LSODA' + newparameternames)
    elif track_RK4:
        J_field_track = np.load('./data/phi_dot_tracking/J_field_RK4' + newparameternames) / scalefactor
        neighbour_track = np.load('./data/phi_dot_tracking/neighbour_RK4' + newparameternames)
        phi_track = np.load('./data/phi_dot_tracking/phi_RK4' + newparameternames)
        print(J_field_track)
        print(phi_track)
        t_track = np.linspace(0.0, cycles, len(J_field_track))

    else:
        J_field_t = np.load('./data/phi_dot_tracking/J_field' + newparameternames) / scalefactor
        neighbour_t = np.load('./data/phi_dot_tracking/neighbour' + newparameternames)
        phi_t = np.load('./data/phi_dot_tracking/phi' + newparameternames)
        J_field_t2 = np.load('./data/phi_dot_tracking/J_field' + newparameternames2) / scalefactor
        neighbour_t2 = np.load('./data/phi_dot_tracking/neighbour' + newparameternames2)
        phi_t2 = np.load('./data/phi_dot_tracking/phi' + newparameternames2)
        energy_t = np.load('./data/phi_dot_tracking/energy' + newparameternames)

        t_track = np.load('./data/phi_dot_tracking/time' + newparameternames)
        t_track2 = np.load('./data/phi_dot_tracking/time' + newparameternames2)
        stepsize = np.load('./data/phi_dot_tracking/stepsize' + newparameternames)
        print(len(J_field_t))
        print(len(t_track))
        J_func = interp1d(t_track, scalefactor * J_field_t, fill_value='extrapolate', bounds_error=False)
        Phi_func = interp1d(t_track, ascale*scalefactor * phi_t.real, fill_value='extrapolate', bounds_error=False)

        for i in range(len(times)):
            phi_track.append(Phi_func(times[i]))
        for i in range(len(times)):
            J_field_track.append(J_func(times[i]))

"""Plot currents"""
plt.subplot(211)
plt.title("$J$ from Evolving with Respect to $\\dot{\\Phi}$")
plt.plot(times, J_field, label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
if Tracking:
    if not track_RK4:
        plt.plot(t_track, J_field_t, linestyle='dashed',
                 label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
    else:
        plt.plot(t_track, J_field_track, linestyle='dashed',
                 label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
#
plt.ylabel('$J(t)$')
plt.legend(loc='upper right')
plt.subplot(212)
plt.plot(times, J_field2, label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F02))
if Tracking:
    if not track_RK4:
        plt.plot(t_track2, J_field_t2, linestyle='dashed',
                 label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F02))
plt.ylabel('$J(t)$')
plt.xlabel('Time [cycles]')
plt.legend(loc='upper right')

plt.savefig('./data/images/J_field_phi_dot_tracked' + imagefilenames)

plt.show()

"""plot phi"""
plt.subplot(211)
plt.title("$\\Phi$ from evolving with $J$")
plt.plot(times, phi_original.real, label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
if Tracking:
    if not track_RK4:
        plt.plot(t_track, phi_t, label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0), linestyle='dashed')
    else:
        plt.plot(t_track, phi_track,
                 label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0),
                 linestyle='dashed')
plt.plot(times, 0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.plot(times, -0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.plot(times, -1.5* np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.plot(times, 1.5 *np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.ylabel('$\\Phi(t)$')
plt.legend(loc='upper right')
plt.subplot(212)
plt.plot(times, phi_original2.real, label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F02))
if Tracking:
    if not track_RK4:
        plt.plot(t_track2, phi_t2, label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F02), linestyle='dashed')
plt.plot(times, 0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.plot(times, -0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')

plt.xlabel('Time [cycles]')
plt.ylabel('$\\Phi(t)$')
plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
plt.legend(loc='upper right')

plt.savefig('./data/images/Phi_phi_dot_tracked' + imagefilenames)

plt.show()


"""plot grad phi"""
plt.plot(times, np.gradient(phi_original, delta), label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
if Tracking:
    if not track_RK4:
        plt.plot(times, np.gradient(phi_track, delta),
                 label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0),
                 linestyle='dashed')
    else:
        plt.plot(t_track, np.gradient(phi_track, delta),
                 label='$\\dot{\\Phi}$ evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0),
                 linestyle='dashed')
plt.ylabel('$\\frac{d \\Phi}{dt}$')
plt.legend(loc='upper right')

plt.show()
if not track_RK4:
    """plotting stepsize"""
    plt.title("Stepsize")
    plt.plot(stepsize)
    plt.ylabel("Step size")
    plt.xlabel("Iteration")
    plt.show()

plt.title("Energy")
plt.plot(times, energy)
plt.plot(t_track, energy_t)
plt.ylabel("Energy")
plt.xlabel("Time")
plt.show()

