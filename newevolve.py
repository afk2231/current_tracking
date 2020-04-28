import numpy as np
import harmonic as ha
import evolve as ev
import mpmath as mp
import definition as de
import observable as ob



def coupled_evolve(t, y, lat, ht, J_dot_funct):
    #
    # y[0] = phi
    # y[1] = R
    # y[2] = theta
    # y[3 through (len(y) - 3)/2 + 3] = psi_r
    # y[(len(y) - 3)/2 + 3 through len(y)] = psi_i
    #
    # phi_dot = [R_dot(psi)/R(psi)tan(theta(psi) - phi) + theta_dot(psi)]
    #               - J_dot / (2*a * t_0)R(psi) cos(theta(psi) - phi)

    psi = recombine_psi(y)
    out = []
    C = 2 * lat.a * lat.t
    h = coupled_h(y, ht)
    Commu = ha.two_body(lat, h, psi.real, psi.imag)
    R_dot = Commu.real*np.cos(y[2]) + Commu.imag*np.sin(y[2])
    theta_dot = (1/y[1])*(Commu.imag*np.cos(y[2]) - Commu.real*np.sin(y[2]))
    diffphase = y[2] - y[0]
    phi_dot = theta_dot - ((R_dot/y[1]) * np.sin(diffphase) + J_dot_funct(t)/(C * y[1]))/np.cos(diffphase)

    out.append(phi_dot.real)
    out.append(R_dot)
    out.append(theta_dot)

    # psi_dot = -i H_hat psi
    psi_dot = ev.f(lat, h, psi)

    # -i H_psi = -i (H_psi_r + i H_psi_i) = -i H_psi_r + H_psi_i = (H_psi_i) + i (-H_psi_r)
    for k in range((len(psi_dot))):
        out.append(psi_dot[k].imag)
    for k in range((len(psi_dot))):
        out.append(-psi_dot[k].real)
    #print(out[2])
    #print(out[3])


    return out


def coupled_evolve_alt(t, y, lat, ht, J_dot_funct, prev_phi_dot, r, time):
    #
    # y[0] = phi
    # y[1] = R
    # y[2] = theta
    # y[3 through (len(y) - 3)/2 + 3] = psi_r
    # y[(len(y) - 3)/2 + 3 through len(y)] = psi_i
    #
    # phi_dot = [R_dot(psi)/R(psi)tan(theta(psi) - phi) + theta_dot(psi)]
    #               - J_dot / (2*a * t_0)R(psi) cos(theta(psi) - phi)

    psi = recombine_psi(y)
    out = []
    C = 2 * lat.a * lat.t
    h = coupled_h(y, ht)
    Commu = ha.two_body(lat, h, psi.real, psi.imag)
    R_dot = Commu.real * np.cos(y[2]) + Commu.imag * np.sin(y[2])
    theta_dot = (1 / y[1]) * (Commu.imag * np.cos(y[2]) - Commu.real * np.sin(y[2]))
    diffphase = y[2] - y[0]
    if r[-1] == 1 and not r[-2] == 1:
        phi_dot = prev_phi_dot[-1] + r[-1] * (1 * prev_phi_dot[-1] - 1 * prev_phi_dot[-2] + 0 * prev_phi_dot[-3])
    elif r[-1] == 1 and r[-2] == 1:
        phi_dot = prev_phi_dot[-1] + (1/2) * r[-1] * (3 * prev_phi_dot[-1] - 4 * prev_phi_dot[-2] + 1 * prev_phi_dot[-3])
    else:
        phi_dot = prev_phi_dot[-1] + r[-1] * (prev_phi_dot[-1] - prev_phi_dot[-2])
    out.append(phi_dot.real)
    out.append(R_dot)
    out.append(theta_dot)

    # psi_dot = -i H_hat psi
    psi_dot = ev.f(lat, h, psi)

    # -i H_psi = -i (H_psi_r + i H_psi_i) = -i H_psi_r + H_psi_i = (H_psi_i) + i (-H_psi_r)
    for k in range((len(psi_dot))):
        out.append(psi_dot[k].imag)
    for k in range((len(psi_dot))):
        out.append(-psi_dot[k].real)
    # print(out[2])
    # print(out[3])

    return out


def recombine_psi(y):
    psi_r = []
    for k in range(3, int((len(y) - 3) / 2 + 3)):
        psi_r.append(y[k])

    psi_i = []
    for k in range(int((len(y) - 3) / 2 + 3), (len(y))):
        psi_i.append(y[k])
    psi = np.zeros(len(psi_r), complex)
    if len(psi_r) == len(psi_i):
        for k in range(len(psi_r)):
            psi[k] = (psi_r[k] + 1j * psi_i[k])
    else:
        print('something went wrong in psi recombination')
        exit(1)
    return psi


def coupled_RK4(t, y, lat, h, J_dot_funct, delta, prev_phi_dot):
    ht = coupled_h(y, h)
    k1 = delta * np.asarray(coupled_evolve(t, y, lat, ht, J_dot_funct))
    ht = coupled_h(np.asarray(y) + 0.5 * k1, h)
    k2 = delta * np.asarray(coupled_evolve(t + 0.5 * delta, np.asarray(y) + 0.5 * k1, lat, ht, J_dot_funct))
    ht = coupled_h(np.asarray(y) + 0.5 * k2, h)
    k3 = delta * np.asarray(coupled_evolve(t + 0.5 * delta, np.asarray(y) + 0.5 * k2, lat, ht, J_dot_funct))
    ht = coupled_h(np.asarray(y) + k3, h)
    k4 = delta * np.asarray(coupled_evolve(t + delta, np.asarray(y) + k3, lat, ht, J_dot_funct))
    out = y + (k1 + 2. * k2 + 2. * k3 + k4)/6.
    prev_phi_dot.append((k1 / delta)[0])
    return out


def coupled_RK4_alt(t, y, lat, h, J_dot_funct, delta, prev_phi_dot, r, time):
    ht = coupled_h(y, h)
    k1 = delta * np.asarray(coupled_evolve_alt(t, y, lat, ht, J_dot_funct, prev_phi_dot, r, time))
    ht = coupled_h(np.asarray(y) + 0.5 * k1, h)
    k2 = delta * np.asarray(coupled_evolve_alt(t + 0.5 * delta, np.asarray(y) + 0.5 * k1, lat, ht, J_dot_funct,
                                               prev_phi_dot, r, time))
    ht = coupled_h(np.asarray(y) + 0.5 * k2, h)
    k3 = delta * np.asarray(coupled_evolve_alt(t + 0.5 * delta, np.asarray(y) + 0.5 * k2, lat, ht, J_dot_funct,
                                               prev_phi_dot, r, time))
    ht = coupled_h(np.asarray(y) + k3, h)
    k4 = delta * np.asarray(coupled_evolve_alt(t + delta, np.asarray(y) + k3, lat, ht, J_dot_funct, prev_phi_dot, r,
                                               time))
    out = y + (k1 + 2. * k2 + 2. * k3 + k4)/6.
    if r[-1] == 1 and not r[-2] == 1:
        prev_phi_dot.append(prev_phi_dot[-1] + r[-1] * (1 * prev_phi_dot[-1] - 1 * prev_phi_dot[-2] + 0 * prev_phi_dot[-3]))
    elif r[-1] == 1 and r[-2] == 1:
        prev_phi_dot.append(prev_phi_dot[-1] + (1 / 2) * r[-1] * (
                    3 * prev_phi_dot[-1] - 4 * prev_phi_dot[-2] + 1 * prev_phi_dot[-3]))
    else:
        prev_phi_dot.append(prev_phi_dot[-1] + r[-1] * (prev_phi_dot[-1] - prev_phi_dot[-2]))
    return out


def coupled_h(y, h):
    h_forwards = np.triu(h)
    h_forwards[0, -1] = 0.0
    h_forwards[-1, 0] = h[-1, 0]
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    return np.exp(1.j * y[0]) * h_forwards + np.exp(-1.j * y[0]) * h_backwards


def prelim_work_coupled_evolve(t, y, lat, ht, jfunct, efunct):
    #
    # y[0] = phi
    # y[1] = R
    # y[2] = theta
    # y[3 through (len(y) - 3)/2 + 3] = psi_r
    # y[(len(y) - 3)/2 + 3 through len(y)] = psi_i
    #
    # phi_dot = [R_dot(psi)/R(psi)tan(theta(psi) - phi) + theta_dot(psi)]
    #               - J_dot / (2*a * t_0)R(psi) cos(theta(psi) - phi)

    psi = recombine_psi(y)
    out = []
    C = 2 * lat.a * lat.t
    h = coupled_h(y, ht)
    Commu = ha.two_body(lat, h, psi.real, psi.imag)
    R_dot = Commu.real*np.cos(y[2]) + Commu.imag*np.sin(y[2])
    theta_dot = (1/y[1])*(Commu.imag*np.cos(y[2]) - Commu.real*np.sin(y[2]))

    j = jfunct(t)
    e = efunct(t)

    print(e)
    #print(j)


    phi_dot = - lat.a * e * j**-1


    out.append(phi_dot.real)
    out.append(R_dot)
    out.append(theta_dot)

    # psi_dot = -i H_hat psi
    psi_dot = ev.f(lat, h, psi)

    # -i H_psi = -i (H_psi_r + i H_psi_i) = -i H_psi_r + H_psi_i = (H_psi_i) + i (-H_psi_r)
    for k in range((len(psi_dot))):
        out.append(psi_dot[k].imag)
    for k in range((len(psi_dot))):
        out.append(-psi_dot[k].real)
    #print(out[2])
    #print(out[3])


    return out


def prelim_work_coupled_RK4(t, y, lat, h, j, h_dot, delta):
    ht = coupled_h(y, h)
    k1 = delta * np.asarray(prelim_work_coupled_evolve(t, y, lat, ht, j, h_dot))
    ht = coupled_h(np.asarray(y) + 0.5 * k1, h)
    k2 = delta * np.asarray(prelim_work_coupled_evolve(t + 0.5 * delta, np.asarray(y) + 0.5 * k1, lat, ht, j, h_dot))
    ht = coupled_h(np.asarray(y) + 0.5 * k2, h)
    k3 = delta * np.asarray(prelim_work_coupled_evolve(t + 0.5 * delta, np.asarray(y) + 0.5 * k2, lat, ht, j, h_dot))
    ht = coupled_h(np.asarray(y) + k3, h)
    k4 = delta * np.asarray(prelim_work_coupled_evolve(t + delta, np.asarray(y) + k3, lat, ht, j, h_dot))
    out = y + (k1 + 2. * k2 + 2. * k3 + k4)/6.
    return out

