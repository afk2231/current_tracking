import numpy as np
import matplotlib.pyplot as plt
import definition as hams

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
ascale = 1.001

Tracking = True

prop = hams.hhg(field=field, nup=number, ndown=0, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')

parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)

J_field = np.load('./data/original/Jfield' + parameternames)
phi_original = np.load('./data/original/phi' + parameternames)
neighbour = np.load('./data/original/neighbour' + parameternames)
times = np.linspace(0.0, cycles, len(J_field))

if Tracking:
    prop_track = hams.hhg(field=field, nup=number, ndown=0, nx=nx, ny=0, U=U_track, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)
    imagefilenames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.png' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)

    J_field_track = np.load('./data/tracking/Jfield_original' + newparameternames) / scalefactor
    phi_raw = np.load('./data/tracking/phi_raw' + newparameternames)
    phi_track = np.load('./data/tracking/phi_original' + newparameternames)
    theta = np.load('./data/tracking/theta_raw' + newparameternames)
    neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
    J_field_track_ap = np.load('./data/tracking/Jfield_original_ap' + newparameternames) / scalefactor
    phi_raw_ap = np.load('./data/tracking/phi_raw_ap' + newparameternames)
    phi_track_ap = np.load('./data/tracking/phi_original_ap' + newparameternames)
    theta_ap = np.load('./data/tracking/theta_raw_ap' + newparameternames)
    neighbour_track_ap = np.load('./data/tracking/neighbour_ap' + newparameternames)

    t_track = np.linspace(0.0, cycles, len(J_field_track))

"""Plot currents"""
plt.title("$J$ from Evolving with Respect to $J$")
plt.plot(times, J_field, label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
if Tracking:
    plt.plot(t_track * prop_track.freq / prop.freq, J_field_track, linestyle='dashed',
             label='Current evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.legend(loc='upper right')

plt.savefig('./data/images/J_field_tracked' + imagefilenames)

plt.show()

"""plot phi"""
plt.title("$\\Phi$ from evolving with $J$")
plt.plot(times, phi_original.real, label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
if Tracking:
    plt.plot(t_track, ascale*phi_track.real, label='Current evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0), linestyle='dashed')
plt.plot(times, 0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.plot(times, -0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.ylabel('$\\Phi(t)$')
plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
plt.legend(loc='upper right')

plt.savefig('./data/images/Phi_field_tracked' + imagefilenames)

plt.show()
"""plot grad(Phi)"""
plt.title("$\\frac{d\\Phi}{dt}$ from evolving with $J$")
plt.plot(times, np.gradient(phi_original.real, delta), label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
if Tracking:
    plt.plot(t_track, np.gradient(ascale*phi_track.real, delta), label='Current evolution with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0), linestyle='dashed')
plt.plot(times, 0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.plot(times, -0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
plt.ylabel('$\\frac{d\\Phi}{dt}(t)$')
plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
plt.legend(loc='upper right')

plt.savefig('./data/images/grad_Phi_field_tracked' + imagefilenames)

plt.show()


if Tracking:
    """plot currents found by RK4 and adaptive methods"""
    plt.title("$J$ from RK4 and Adaptive methods")
    plt.plot(times, J_field, label='Original $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
    plt.plot(t_track * prop_track.freq / prop.freq, J_field_track, label='RK4 with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
    print(len(J_field_track_ap))
    plt.plot(t_track * prop_track.freq / prop.freq, J_field_track_ap, linestyle='dashed', label='Adaptive with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
    # plt.xlabel('Time [cycles]')
    plt.ylabel('$J(t)$')
    plt.legend(loc='upper right')
    plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)

    plt.savefig('./data/images/J_field_RK4vsAdap' + imagefilenames)

    plt.show()

    """plot phi found by RK4 and adaptive methods"""
    plt.title("$\\Phi$ from RK4 and Adaptive methods")
    plt.plot(times, phi_original.real,
             label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0))
    plt.plot(t_track, ascale * phi_track.real, label='RK4 with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$'
                                                     % (prop_track.U, F0), linestyle='dotted')
    plt.plot(t_track, ascale * phi_track_ap.real, label='Adaptive with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$'
                                                        % (prop_track.U, F0), linestyle='dashed')
    plt.plot(times, 0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
    plt.plot(times, -0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
    plt.ylabel('$\\Phi(t)$')
    plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
               [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
    plt.legend(loc='upper right')

    plt.savefig('./data/images/Phi_field_tracked_RK4vsAdap' + imagefilenames)

    plt.show()
    """plot grad(Phi)"""
    plt.title("$\\frac{d\\Phi}{dt}$ from evolving with $J$")
    plt.plot(times, np.gradient(phi_original.real, delta), label='Original with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$'
                                                     % (prop_track.U, F0))
    plt.plot(t_track, np.gradient(ascale * phi_track.real, delta),
             label='RK4 with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0), linestyle='dotted')
    plt.plot(t_track, ascale * np.gradient(ascale * phi_track_ap.real, delta),
             label='Adaptive with $\\frac{U}{t_0}=%.1f$ and $F_0 = %s$' % (prop_track.U, F0),
             linestyle='dashed')
    plt.plot(times, 0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
    plt.plot(times, -0.5 * np.pi * np.ones(len(times)), linestyle='dotted', color='red')
    plt.ylabel('$\\frac{d\\Phi}{dt}(t)$')
    plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
               [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
    plt.legend(loc='upper right')

    plt.savefig('./data/images/grad_Phi_field_tracked_RK4vsAdap' + imagefilenames)

    plt.show()