import numpy as np
import matplotlib.pyplot as plt
import definition as hams
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


def analytic_energies(Phi, lat):
    E = 0
    n_plus = lat.nup
    if n_plus % 2 == 1:
        E += -2 * lat.t * np.cos(Phi)
        for k in range(1, int((n_plus - 1) / 2)):
            E += -4 * lat.t * np.cos((2 * np.pi * k) / lat.nx) * np.cos(Phi)
    else:
        E += - 2 * lat.t * np.cos(Phi) - 2 * lat.t * np.cos((np.pi * n_plus) / lat.nx) * np.cos(Phi)
        for k in range(1, int(n_plus / 2 - 1)):
            E += -4 * lat.t * np.cos(((2 * np.pi * k) / lat.nx)) * np.cos(Phi)
    n_plus = lat.ndown

    if n_plus % 2 == 1:
        E += -2 * lat.t * np.cos(Phi)
        for k in range(1, int((n_plus - 1) / 2)):
            E += -4 * lat.t * np.cos((2 * np.pi * k) / lat.nx) * np.cos(Phi)
    else:
        E += - 2 * lat.t * np.cos(Phi) - 2 * lat.t * np.cos((np.pi * n_plus) / lat.nx) * np.cos(Phi)
        for k in range(1, int(n_plus / 2 - 1)):
            E += -4 * lat.t * np.cos(((2 * np.pi * k) / lat.nx)) * np.cos(Phi)
    return E


def analytic_energies_alt(Phi, lat):
    E = 0
    phi_0 = 0.000001
    n = lat.nup
    if n % 2 == 0:
        E += - 2 * lat.t * np.cos(Phi) - 2 * lat.t * np.cos((np.pi * n / lat.nx)) * np.cos(Phi) - 2 * lat.t * \
             np.sin((np.pi * n / lat.nx)) * np.cos(Phi)
        for k in range(1, int(n / 2 - 1)):
            E += -4 * lat.t * np.cos(((2 * np.pi * k) / lat.nx))
    else:
        E += -2 * lat.t * np.cos(Phi)
        for k in range(1, int((n - 1) / 2)):
            E += -4 * lat.t * np.cos((2 * np.pi * k) / lat.nx)

    n = lat.ndown
    if n % 2 == 0:
        E += - 2 * lat.t * np.cos(Phi) - 2 * lat.t * np.cos((np.pi * n / lat.nx)) * np.cos(Phi) - 2 * lat.t * \
             np.sin((np.pi * n / lat.nx)) * np.cos(Phi)
        for k in range(1, int(n / 2 - 1)):
            E += -4 * lat.t * np.cos(((2 * np.pi * k) / lat.nx))
    else:
        E += -2 * lat.t * np.cos(Phi)
        for k in range(1, int((n - 1) / 2)):
            E += -4 * lat.t * np.cos((2 * np.pi * k) / lat.nx)
    return E

def phi_guess(phi, lat):
    E = -(11.7/2)* lat.t * np.cos(phi)
    return E

number = 2
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
ascale = 1.0001
ascale_phi_dot = 1.0
delta_phi_dot = 0.01

prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')

parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)
phi_dot_parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
    nx, cycles, U, t, number, delta_phi_dot, field, F0, ascale_phi_dot, scalefactor)
energy = np.load('./data/original/energy' + parameternames)
phi = np.load('./data/original/phi' + parameternames)
j = np.load('./data/tracking/Jfield_original' + newparameternames)
grad_h = np.gradient(energy, delta)
phi_t = np.load('./data/tracking/phi_original' + newparameternames)
energy_track = np.load('./data/tracking/energies' + newparameternames)
energy_phi_dot = np.load('./data/phi_dot_tracking/energy' + phi_dot_parameternames)
times_phi_dot = np.load('./data/phi_dot_tracking/time' + phi_dot_parameternames)
times = np.linspace(0, cycles, len(energy))
times_t = np.linspace(0, cycles, len(energy_track))
grad_phi = np.gradient(phi_t, delta)
therm_e = j * grad_phi / prop.a

E_pd = interp1d(times_phi_dot, energy_phi_dot, fill_value='extrapolate', bounds_error=False,
                          kind='cubic')
energy_pd = []
for i in range(len(times_t)):
    energy_pd.append(E_pd(times_t[i]))

an_energy = analytic_energies(phi, prop)

"""plot energies"""
plt.title('Energies with Old Tracking Method')
plt.plot(times, energy, label='$\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.plot(times_t, energy_track, linestyle='dashed', label='$\\left\\langle\\hat{H}_T\\right\\rangle(t)$')
plt.plot(times, an_energy, linestyle='dotted', label='Analytic $\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.plot(times, np.zeros(len(times)), linestyle='dotted', color='red')
plt.xlabel('Time (s)')
plt.ylabel('$\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.legend(loc='upper right')

plt.show()


"""plot energies"""
plt.title('Energies with Old Tracking Method')
plt.plot(times_t, energy_track, linestyle='dashed', label='$\\left\\langle\\hat{H}_T\\right\\rangle(t)$')
plt.plot(times, an_energy, linestyle='dotted', label='Analytic $\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.plot(times, np.zeros(len(times)), linestyle='dotted', color='red')
plt.xlabel('Time (s)')
plt.ylabel('$\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.legend(loc='upper right')

plt.show()


plt.plot(times, np.gradient(energy, times))
plt.plot(times_t, np.gradient(energy_track, times_t), linestyle='dashed')

plt.show()

plt.title('Energies with New Tracking Method')
plt.plot(times, energy, label='$\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.plot(times_t, energy_pd, linestyle='dashed', label='$\\left\\langle\\hat{H}_T\\right\\rangle(t)$')
plt.plot(times, an_energy, linestyle='dotted', label='Analytic $\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.plot(times, np.zeros(len(times)), linestyle='dotted', color='red')

plt.xlabel('Time (s)')
plt.ylabel('$\\left\\langle\\hat{H}\\right\\rangle(t)$')
plt.legend(loc='upper right')

plt.show()

plt.plot(times_t, energy_track)
plt.plot(times_t[1:], cumtrapz(-therm_e, dx=delta) + an_energy[0], linestyle='dotted')

plt.show()
