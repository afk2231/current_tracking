import newevolve as nev
import numpy as np
import harmonic as ha
import evolve as ev


def evolve_pole_vaulting(t, y, time, step_size, lat, h, j_dot, n, ndelta):
    y = y + ndelta * np.asarray(nev.coupled_evolve(t, y, lat, h, j_dot))
    step_size.append(ndelta)
    time.append(t * lat.freq)
    if time[-1] < time[-2]:
        exit("fuck")
    return y


def coupled_RK2(t, y, lat, h, j_dot, delta):
    ht = nev.coupled_h(y, h)
    k1 = delta * np.asarray(nev.coupled_evolve(t, y, lat, ht, j_dot))
    ht = nev.coupled_h(y + 0.5 * k1, h)
    k2 = delta * np.asarray(nev.coupled_evolve(t + 0.5 * delta, y + 0.5 * k1, lat, ht, j_dot))
    out = y + k2
    return out
