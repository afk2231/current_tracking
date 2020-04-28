import os
from main_tracking import main_tracking_phi
import multiprocessing as mp


def mpro_phi(u, de, asc, f0):
    os.environ["OMP_NUM_THREADS"] = "%s" % 2
    main_tracking_phi(3, 6, u, de, asc, f0)


if __name__ == "__main__":
    Us = [0, 0.1]
    deltas = [0.05, 0.01]
    ascales = [1.001, 1.01, 1.1]
    F0s = [10]
    jobs = []
    njobs = 0
    print("Starting multiprocessing procedure")
    njobs = 0
    for k in range(len(Us)):
        for el in range(len(deltas)):
            for m in range(len(ascales)):
                U = Us[k]
                delt = deltas[el]
                ascale = ascales[m]
                process2 = mp.Process(target=mpro_phi, args=[U, delt, ascale])
                njobs += 3
                if njobs > 72:
                    print("Max number of cores used")
                    break

                process2.start()
                jobs.append(process2)

    for j in jobs:
        j.join()

    print("Tracking Sim finished.")